import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class Attention(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(
            encoder_dim,
            attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(
            decoder_dim,
            attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(
            attention_dim,
            1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """
    def __init__(self,
                 attention_dim,
                 embed_dim,
                 decoder_dim,
                 vocab_size,
                 encoder_dim=512,
                 dropout=0.5,
                 args=None):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.sample_temp = 0.5

        self.attention = Attention(encoder_dim, decoder_dim,
                                   attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim,
                                       decoder_dim,
                                       bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(
            encoder_dim,
            decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(
            decoder_dim,
            encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(
            decoder_dim,
            vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights(
        )  # initialize some layers with the uniform distribution
        self.args = args

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self,
                encoder_out,
                encoded_captions,
                caption_lengths,
                scheduled_sampling=False,
                decay_schedule=0.5):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(
            batch_size, -1,
            encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(0).sort(
            dim=0, descending=True)
        if batch_size != 1:
            encoder_out = encoder_out[sort_ind]
            encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths).tolist()
        if batch_size == 1:
            decode_lengths = [decode_lengths]
        # decode_lengths = self.args.decode_lengths
        seq_len = encoded_captions.size(1)
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths),
                                  vocab_size).cuda()
        alphas = torch.zeros(batch_size, max(decode_lengths),
                             num_pixels).cuda()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding

        if scheduled_sampling:
            use_sampling = np.random.random() < decay_schedule
        else:
            use_sampling = False

        input_word = torch.ones(batch_size, 1,
                                dtype=torch.long).cuda()  #ones means start
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            if use_sampling:
                word_embed = self.embedding(input_word).squeeze(1)
                word_embed = word_embed[:batch_size_t, :]
            else:
                word_embed = embeddings[:batch_size_t, t, :]
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t])
                                )  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([word_embed, attention_weighted_encoding], dim=1),
                (h[:batch_size_t],
                 c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            if use_sampling:
                scaled_output = preds / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                input_word = scoring.topk(1)[1]

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def forward(self,
                encoder_out,
                encoded_captions,
                caption_lengths,
                use_sampling=False,
                data_parallel=False,
                lstm_criterion=None,
                args=None):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(
            batch_size, -1,
            encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(0).sort(
            dim=0, descending=True)
        if batch_size != 1:
            encoder_out = encoder_out[sort_ind]
            encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths).tolist()
        if batch_size == 1:
            decode_lengths = [decode_lengths]
        # decode_lengths = self.args.decode_lengths
        seq_len = encoded_captions.size(1)
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths),
                                  vocab_size).cuda()
        alphas = torch.zeros(batch_size, max(decode_lengths),
                             num_pixels).cuda()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding

        input_word = torch.ones(batch_size, 1,
                                dtype=torch.long).cuda()  #ones means start
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            if use_sampling:
                word_embed = self.embedding(input_word).squeeze(1)
                word_embed = word_embed[:batch_size_t, :]
            else:
                word_embed = embeddings[:batch_size_t, t, :]
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t])
                                )  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([word_embed, attention_weighted_encoding], dim=1),
                (h[:batch_size_t],
                 c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            if use_sampling:
                scaled_output = preds
                scoring = F.log_softmax(scaled_output, dim=1)
                input_word = scoring.topk(1)[1]
        if not data_parallel:
            return predictions, encoded_captions, decode_lengths, alphas, sort_ind
        else:
            targets = encoded_captions[:, 1:]
            args.scores_copy = predictions.clone()
            args.targets_copy = targets.clone()
            args.decode_lengths = decode_lengths
            scores, *_ = pack_padded_sequence(predictions,
                                              decode_lengths,
                                              batch_first=True)
            targets, *_ = pack_padded_sequence(targets,
                                               decode_lengths,
                                               batch_first=True)

            loss = lstm_criterion(scores, targets)
            loss += args.alpha_c * ((1. - alphas.sum(dim=1))**2).mean()
            return loss

    def greedy_search(self, features, max_sentence=10):
        batch_size = features.size(0)
        encoder_dim = features.size(1)
        features = features.view(
            batch_size, -1,
            encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = features.size(1)
        predictions = torch.zeros(batch_size, max_sentence,
                                  self.vocab_size).cuda()
        alphas = torch.zeros(batch_size, max_sentence, num_pixels).cuda()
        sentence = torch.zeros(batch_size, max_sentence).cuda()
        input_word = torch.ones(batch_size, 1, dtype=torch.long).cuda()
        h, c = self.init_hidden_state(features)
        h_pred = torch.zeros(batch_size, max_sentence, self.decoder_dim).cuda()
        c_pred = torch.zeros(batch_size, max_sentence, self.decoder_dim).cuda()
        step = 0
        while True:
            word_embed = self.embedding(input_word).squeeze(1)
            attention_weighted_encoding, alpha = self.attention(features, h)
            gate = self.sigmoid(
                self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([word_embed, attention_weighted_encoding], dim=1),
                (h, c))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))
            scoring = F.log_softmax(preds, dim=1)
            top_idx = scoring.topk(1)[1]
            input_word = top_idx
            sentence[:, step] = top_idx.squeeze(1)
            predictions[:, step, :] = preds
            alphas[:, step, :] = alpha
            h_pred[:, step, :] = h
            c_pred[:, step, :] = c
            step += 1
            if (step >= max_sentence):
                break
        return sentence, alphas, predictions, h_pred, c_pred

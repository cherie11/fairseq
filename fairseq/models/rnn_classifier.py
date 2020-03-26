import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models import register_model_architecture
from fairseq import options, utils
import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models.lstm import LSTMEncoder
# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.
def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx)
    embed_dict = utils.parse_embedding(embed_path)
    utils.print_embed_overlap(embed_dict, dictionary)
    return utils.load_embedding(embed_dict, dictionary, embed_tokens)
DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5
class RNNClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@register_model('rnn_classifier')
class FairseqRNNClassifier(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add a new command-line argument to configure the
        # dimensionality of the hidden state.
        parser.add_argument(
            '--hidden-dim', type=int, metavar='N',
            help='dimensionality of the hidden state',
        )
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension',default=300)
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='encoder embedding dimension',default=2)
        parser.add_argument('--encoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-dropout-in',  type=float, default=0.2)
        parser.add_argument('--encoder-dropout-out',  type=float, default=0.2)
        
        
    
    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a FairseqRNNClassifier instance.
        max_source_positions = getattr(args, 'max_source_positions', DEFAULT_MAX_SOURCE_POSITIONS)
        max_target_positions = getattr(args, 'max_target_positions', DEFAULT_MAX_TARGET_POSITIONS)

        # Initialize our RNN module        
        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )
            
        
        
        rnn = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=True,
            pretrained_embed=pretrained_encoder_embed,
        )
        output_size = 25
        last_hidden = nn.Linear(args.hidden_dim*2*args.encoder_layers, output_size)
        # Return the wrapped version of the module
        return FairseqRNNClassifier(
            rnn=rnn,
            last_hidden=last_hidden,
            input_vocab=task.source_dictionary,
        )
    def __init__(self, rnn, input_vocab,last_hidden,**kwargs):
        super(FairseqRNNClassifier, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.last_hidden = last_hidden
        self.rnn = rnn
        self.input_vocab = input_vocab
        self.classification_heads=dict()
        self.classification_heads['sentence_classification_head']=1
        # The RNN module in the tutorial expects one-hot inputs, so we can
        # precompute the identity matrix to help convert from indices to
        # one-hot vectors. We register it as a buffer so that it is moved to
        # the GPU when ``cuda()`` is called.
        self.register_buffer('one_hot_inputs', torch.eye(len(input_vocab)))
    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        pass
    def forward(self, src_tokens, src_lengths,*args,**kwargs):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch. We'll define the Task in the next section, but for
        # now just know that *src_tokens* has shape `(batch, src_len)` and
        # *src_lengths* has shape `(batch)`.

        x, final_hiddens, final_cells = self.rnn(src_tokens.to(torch.long), src_lengths=src_lengths)['encoder_out']
        final_hiddens = torch.reshape(final_hiddens,(final_hiddens.size()[1],-1))
       
        output = self.last_hidden(final_hiddens)
        # output = self.softmax(output)
        # Return the final output state for making a prediction
        return output,1
    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return 2

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'rnn_classifier'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.

@register_model_architecture('rnn_classifier', 'pytorch_tutorial_rnn')
def pytorch_tutorial_rnn(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.hidden_dim = getattr(args, 'hidden_dim', 1024)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path')
        
    
    
    
    
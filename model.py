#!/usr/bin/env python
# coding: utf-8

# In[1]:



import tensorflow as tf
import tensorflow.keras as keras

# In[2]:


class AsymmetricConvolution(tf.keras.Model):
    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()
        
        self.x1 = keras.layers.ZeroPadding2D(padding = (1, 0), data_format = "channels_first")
        self.conv1 = keras.layers.Conv2D(filters = out_cha, kernel_size=(3, 1), use_bias=False, data_format = "channels_first")
        
        self.x2 = keras.layers.ZeroPadding2D(padding = (0, 1), data_format = "channels_first")
        self.conv2 = keras.layers.Conv2D(filters = out_cha, kernel_size=(1, 3), data_format = "channels_first")

        self.shortcut = lambda x: x

        if in_cha != out_cha:
            self.shortcut = keras.layers.Conv2D(out_cha, kernel_size = 1, use_bias=False, data_format = "channels_first")
    
        self.activation = keras.layers.ReLU()

    def call(self, x):

        shortcut = self.shortcut(x)
        
        x1 = self.x1(x)       
        x1 = self.conv1(x1)
        
        x2 = self.x2(x)
        x2 = self.conv2(x2)
        
        x2 = self.activation(x2 + x1)    

        return x2 + shortcut

class InteractionMask(tf.keras.Model):

    def __init__(self, number_asymmetric_conv_layer=7, spatial_channels=4, temporal_channels=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = keras.Sequential()
        self.temporal_asymmetric_convolutions = keras.Sequential()


        
        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.add(
                AsymmetricConvolution(spatial_channels, spatial_channels),
            )
            self.temporal_asymmetric_convolutions.add(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = keras.layers.Activation(keras.activations.sigmoid)
        self.temporal_output = keras.layers.Activation(keras.activations.sigmoid)

    def call(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):

        assert len(dense_temporal_interaction.shape) == 4
        assert len(dense_spatial_interaction.shape) == 4
        
        # ----------------------------------------------------------------
        dense_spatial_interaction = self.spatial_asymmetric_convolutions(dense_spatial_interaction)
        dense_temporal_interaction = self.temporal_asymmetric_convolutions(dense_temporal_interaction)
        # ----------------------------------------------------------------
        
        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        spatial_zero = tf.zeros_like(spatial_interaction_mask)
        temporal_zero = tf.zeros_like(temporal_interaction_mask)

        spatial_interaction_mask = tf.where(spatial_interaction_mask > threshold, spatial_interaction_mask,
                                               spatial_zero)

        temporal_interaction_mask = tf.where(temporal_interaction_mask > threshold, temporal_interaction_mask,
                                               temporal_zero)

        return spatial_interaction_mask, temporal_interaction_mask


# In[4]:


class ZeroSoftmax(tf.keras.Model):

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def call(self, x, dim=0, eps=1e-5):
        x_exp = tf.math.pow(tf.math.exp(x) - 1, 2)
        
        x_exp_sum = tf.math.reduce_sum(x_exp, axis=dim, keepdims=True)
        x = x_exp / (x_exp_sum + eps)
        return x


# In[5]:


class SelfAttention(tf.keras.Model):

    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        
        self.embedding = keras.layers.Dense(d_model)
        self.query = keras.layers.Dense(d_model)
        self.key = keras.layers.Dense(d_model)
        
        self.scaled_factor = tf.math.sqrt(tf.convert_to_tensor([d_model], dtype = tf.float64))
        self.softmax = keras.layers.Softmax(axis = -1)

        self.num_heads = num_heads

    def split_heads(self, x):
        
        x = tf.reshape(x, (x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads)) # .contiguous()
    
        x = tf.transpose(x, perm = (0, 2, 1, 3))
        return x

    def call(self, x, mask=False, multi_head=False):

        assert len(x.shape) == 3
        
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = tf.linalg.matmul(query, tf.transpose(key, perm = (0, 1, 3, 2)))  
        else:
            attention = tf.linalg.matmul(query, tf.transpose(key, perm = (0, 1, 3, 2)))  # (batch_size, seq_len, seq_len)
        
        attention = tf.cast(attention, dtype = tf.float64)
        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:

            mask = tf.ones_like(attention)
            attention = attention * tf.experimental.numpy.tril(mask)

        return attention, embeddings


# In[6]:


class SpatialTemporalFusion(tf.keras.Model):

    def __init__(self, obs_len = 8):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = keras.Sequential(
            [
                keras.layers.Conv2D(obs_len, kernel_size = 1, data_format = "channels_first", padding = 'same'),
                keras.layers.ReLU()
            ]
        )

        self.shortcut = keras.Sequential()

    def call(self, x):

        x = self.conv(x) + self.shortcut(x)

        return tf.squeeze(x)


# In[7]:


class SparseWeightedAdjacency(tf.keras.Model):

    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, obs_len=8, dropout=0,
                 number_asymmetric_conv_layer=7):
        super(SparseWeightedAdjacency, self).__init__()

        # dense interaction
        self.spatial_attention = SelfAttention(in_dims = spa_in_dims, d_model = embedding_dims)
        self.temporal_attention = SelfAttention(in_dims = tem_in_dims, d_model = embedding_dims)

        # attention fusion
        self.spa_fusion = SpatialTemporalFusion(obs_len = obs_len)

        # interaction mask
        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer = number_asymmetric_conv_layer
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()

    def call(self, graph, identity):

        assert len(graph.shape) == 3
        
        spatial_graph = graph[:, :, 1:]  # (T N 2)
        #temporal_graph = Permute(dims = (2, 1, 3), input_shape = (8, None, 3))(graph)  # (N T 3)
        temporal_graph = tf.transpose(graph, perm = [1, 0, 2])
        
        # (T num_heads N N)   (T N d_model)
        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True)

        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph, multi_head=True)

        # attention fusion
        st_interaction = tf.transpose(self.spa_fusion(tf.transpose(dense_spatial_interaction, perm = (1, 0, 2, 3))), perm = (1, 0, 2, 3))
        ts_interaction = dense_temporal_interaction

        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)

        # self-connected
        spatial_mask = spatial_mask + tf.expand_dims(identity[0], axis = 1)
        temporal_mask = temporal_mask + tf.expand_dims(identity[1], axis = 1)

        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)
        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings


# In[8]:


class GraphConvolution(tf.keras.Model):

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = keras.layers.Dense(embedding_dims, use_bias=False)
        self.activation = keras.layers.ReLU()

        self.dropout = dropout

    def call(self, graph, adjacency):

        gcn_features = self.embedding(tf.linalg.matmul(adjacency, graph))

        gcn_features = tf.keras.layers.Dropout(rate = self.dropout)(self.activation(gcn_features))

        return gcn_features  # [batch_size num_heads seq_len hidden_size]


# In[9]:


class SparseGraphConvolution(tf.keras.Model):

    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn_1 = GraphConvolution(in_dims, embedding_dims)
        self.spatial_temporal_sparse_gcn_2 = GraphConvolution(embedding_dims, embedding_dims)

        self.temporal_spatial_sparse_gcn_1 = GraphConvolution(in_dims, embedding_dims)
        self.temporal_spatial_sparse_gcn_2 = GraphConvolution(embedding_dims, embedding_dims)

    def call(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):

        graph = graph[:, :, :, 1:]
        spa_graph = tf.transpose(graph, perm = (1, 0, 2, 3))  # (seq_len 1 num_p 2)
        tem_graph = tf.transpose(spa_graph, perm = (2, 1, 0, 3))  # (num_p 1 seq_len 2)
        
        gcn_spatial_features = self.spatial_temporal_sparse_gcn_1(spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = tf.transpose(gcn_spatial_features, perm = (2, 1, 0, 3))

        gcn_spatial_temporal_features = self.spatial_temporal_sparse_gcn_2(gcn_spatial_features, normalized_temporal_adjacency_matrix)
        
        gcn_temporal_features = self.temporal_spatial_sparse_gcn_1(tem_graph, normalized_temporal_adjacency_matrix)
        gcn_temporal_features = tf.transpose(gcn_temporal_features, perm = (2, 1, 0, 3))
        gcn_temporal_spatial_features = self.temporal_spatial_sparse_gcn_2(gcn_temporal_features, normalized_spatial_adjacency_matrix)

        return gcn_spatial_temporal_features, tf.transpose(gcn_temporal_spatial_features, perm = (2, 1, 0, 3))


# In[10]:


class TrajectoryModel(tf.keras.Model):

    def __init__(self,
                 number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len = 12, n_tcn=5,
                 out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer
        )

        # graph convolution
        self.stsgcn = SparseGraphConvolution(
            in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.fusion_ = keras.layers.Conv2D(num_heads, kernel_size=1, use_bias=False, data_format = "channels_first")

        # self.tcns = keras.Sequential()
        self.tcns = []
        self.tcns.append(keras.Sequential(
            [
                keras.layers.Conv2D(filters = pred_len, kernel_size = 3, data_format = "channels_first", padding="same"),
                keras.layers.ReLU()
            ]
        ))

        for j in range(1, self.n_tcn):
            self.tcns.append(keras.Sequential(
                [
                    keras.layers.Conv2D(filters = pred_len, kernel_size = 3, data_format = "channels_first", padding="same"),
                    keras.layers.ReLU()
                ] 
            ))

        self.outputs = keras.layers.Dense(out_dims) # input dim = embedding_dims // num_heads, 我不确定要不要用到

    def call(self, graph, identity):

        # graph 1 obs_len N 3

        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings =             self.sparse_weighted_adjacency_matrices(tf.squeeze(graph), identity)

        gcn_temporal_spatial_features, gcn_spatial_temporal_features = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )


        gcn_representation = self.fusion_(gcn_temporal_spatial_features) + gcn_spatial_temporal_features

        gcn_representation = tf.transpose(gcn_representation, perm = (0, 2, 1, 3))

        features = self.tcns[0](gcn_representation)
        
        for k in range(1, self.n_tcn):
            features = tf.keras.layers.Dropout(rate = self.dropout)(self.tcns[k](features) + features)

        prediction = tf.math.reduce_mean(self.outputs(features), axis=-2)

        return tf.transpose(prediction, perm = (1, 0, 2))


import tensorflow as tf
from tensorflow.keras import layers, initializers
import math

@tf.keras.utils.register_keras_serializable(package="eeg")
class MambaBlock(layers.Layer):
    """
    Mamba Block implementation in TensorFlow/Keras.
    
    Ref: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" by Albert Gu, Tri Dao.
    This implementation uses tf.scan for the selective state space recurrence, 
    which is an approximation suitable for graph execution but slower than custom CUDA kernels.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", conv_bias=True, bias=False, dropout=0.0, **kwargs):
        """
        Args:
            d_model: Input embedding dimension.
            d_state: SSM state dimension (N in paper).
            d_conv: Convolution kernel size.
            expand: Expansion factor for inner dimension.
            dt_rank: Rank of delta projection. "auto" means ceil(d_model / 16).
            conv_bias: Whether to use bias in Conv1D.
            bias: Whether to use bias in linear projections.
            dropout: Dropout rate.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.conv_bias = conv_bias
        self.bias = bias
        self.dropout_rate = dropout
        
        if dt_rank == "auto":
            self.bias = bias
        
        if dt_rank == "auto":
            self.dt_rank = int(math.ceil(self.d_model / 16))
        else:
            self.dt_rank = dt_rank

    def build(self, input_shape):
        # 1. Input Projection: x -> [x_branch, z_branch]
        # Maps D -> 2*ED
        self.in_proj = layers.Dense(self.d_inner * 2, use_bias=self.bias, name='in_proj')
        
        # 2. Conv1D for x_branch
        # Use DepthwiseConv2D to strictly enforce depthwise behavior and avoid Conv2DBackpropInput shape errors with groups
        # We will reshape input to (B, L, 1, D) and use kernel (K, 1)
        self.conv1d = layers.DepthwiseConv2D(
            kernel_size=(self.d_conv, 1),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            use_bias=self.conv_bias,
            name='conv1d'
        )
        
        # 3. SSM Parameters Projections
        # x_proj takes x_branch and projects to [dt, B, C]
        # Dimensions: dt_rank + d_state + d_state
        self.x_proj = layers.Dense(self.dt_rank + self.d_state * 2, use_bias=False, name='x_proj')
        
        # dt_proj project dt_rank to d_inner
        self.dt_proj = layers.Dense(self.d_inner, use_bias=True, name='dt_proj')
        
        # A parameter: (d_inner, d_state)
        # Initialized to preserve signal (often using HiPPO-like structure or just random)
        # We use a random uniform processed by exp later to ensure negative eigenvalues
        self.A_log = self.add_weight(
            name='A_log',
            shape=(self.d_inner, self.d_state),
            initializer=initializers.RandomUniform(minval=math.log(1), maxval=math.log(10)),
            trainable=True
        )
        
        # D parameter: (d_inner,)
        # Skip connection for the SSM
        self.D = self.add_weight(
            name='D',
            shape=(self.d_inner,),
            initializer='ones',
            trainable=True
        )
        
        # 4. Out Projection
        self.out_proj = layers.Dense(self.d_model, use_bias=self.bias, name='out_proj')
        
        self.dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        inputs: (Batch, Length, Dim)
        """
        # 1. Projection
        xz = self.in_proj(inputs) # (B, L, 2*ED)
        x, z = tf.split(xz, 2, axis=-1)
        
        # 2. Conv & SiLU (SSM Input formulation)
        # Reshape for DepthwiseConv2D: (B, L, D) -> (B, L, 1, D)
        x_reshaped = tf.expand_dims(x, axis=2)
        x_conv = self.conv1d(x_reshaped)
        # Back to (B, L, D)
        x = tf.squeeze(x_conv, axis=2)
        
        x = tf.nn.silu(x)
        
        # 3. SSM (Selective Scan)
        # Compute discrete time-step properties from input x
        
        # x_dbl: (B, L, dt_rank + 2*N)
        x_dbl = self.x_proj(x)
        dt_in, B, C = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=-1)
        
        # dt: (B, L, ED)
        dt = self.dt_proj(dt_in)
        dt = tf.nn.softplus(dt) # Softplus to ensure positive Delta
        
        # A: (ED, N)
        # Using -exp(A_log) to ensure diagonal A is strictly negative (Stability)
        A = -tf.exp(self.A_log)
        
        # Prepare for scan: Transpose to (L, B, ...) as tf.scan iterates on axis 0
        x_t = tf.transpose(x, [1, 0, 2])     # (L, B, ED)
        dt_t = tf.transpose(dt, [1, 0, 2])   # (L, B, ED)
        B_t = tf.transpose(B, [1, 0, 2])     # (L, B, N)
        C_t = tf.transpose(C, [1, 0, 2])     # (L, B, N)
        
        # Initial State: (B, ED, N)
        batch_size = tf.shape(inputs)[0]
        initial_state = tf.zeros((batch_size, self.d_inner, self.d_state), dtype=inputs.dtype)
        
        # Scan function for recurrence: h_t = A_bar * h_{t-1} + B_bar * x_t
        def scan_step(h_prev, input_tuple):
            xt, dtt, Bt, Ct = input_tuple
            
            # Discretize A: A_bar = exp(dt * A)
            # dtt: (B, ED), A: (ED, N) -> dA: (B, ED, N)
            dA = tf.exp(tf.einsum('bd,dn->bdn', dtt, A))
            
            # Discretize B: B_bar = dt * B (Simplified Zero-Order Hold)
            # dtt: (B, ED), Bt: (B, N) -> dB: (B, ED, N)
            dB = tf.einsum('bd,bn->bdn', dtt, Bt)
            
            # Update State
            # h_prev: (B, ED, N)
            # xt: (B, ED) -> expand to (B, ED, 1)
            # h_new = dA * h_prev + dB * xt
            h_new = dA * h_prev + dB * tf.expand_dims(xt, axis=-1)
            
            return h_new

        # Run Scan
        # Returns sequence of states h: (L, B, ED, N)
        h_seq = tf.scan(
            scan_step,
            elems=(x_t, dt_t, B_t, C_t),
            initializer=initial_state
        )
        
        # 4. Compute SSM Output
        # Transpose back to (B, L, ED, N)
        h_seq = tf.transpose(h_seq, [1, 0, 2, 3])
        
        # y = h * C
        # h: (B, L, ED, N), C: (B, L, N) -> Output y: (B, L, ED)
        y = tf.einsum('bldn,bln->bld', h_seq, C)
        
        # Add skip connection D * x
        y = y + x * self.D
        
        # 5. Gating
        out = y * tf.nn.silu(z)
        
        # 6. Output Projection
        out = self.out_proj(out)
        
        out = self.dropout(out, training=training)
        
        return out

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model,
                "d_state": self.d_state, "d_conv": self.d_conv,
                "expand": self.expand, "dt_rank": self.dt_rank,
                "conv_bias": self.conv_bias, "bias": self.bias,
                "dropout": self.dropout_rate}

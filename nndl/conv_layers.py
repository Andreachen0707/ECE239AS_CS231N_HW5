import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_o = 1 + (H + 2 * pad - HH) // stride
  W_o = 1 + (W + 2 * pad - WW) // stride
  window_input=np.zeros((N,F,C,H_o,W_o))
  out = np.zeros((N,F,H_o,W_o))
  def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

  for i in range(0,N):
    for f in range(0,F):
      for m in range(0,H_o):
        hs = m*stride
        for n in range(0,W_o):
          ws = n*stride
          for c in range(0,C):
            x_pad = np.pad(x[i,c,:,:],pad,pad_with)
            window_input[i,f,c,m,n] += np.sum(x_pad[hs:hs+HH,ws:ws+WW]*w[f,c,:,:])
          out[i,f,m,n] = np.sum(window_input[i,f,:,m,n])
      out[i,f,:,:]+=b[f]


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  x,w,b,conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_o = 1 + (H + 2 * pad - HH) // stride
  W_o = 1 + (W + 2 * pad - WW) // stride
    
  x_pad = np.zeros((N,C,H+2*pad,W+2*pad))
  dx_pad = np.zeros(x_pad.shape)
  dx = np.zeros(x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
    
  def pad_with(vector, pad_width, iaxis, kwargs):
       pad_value = kwargs.get('padder', 0)
       vector[:pad_width[0]] = pad_value
       vector[-pad_width[1]:] = pad_value
       return vector

  for i in range(0,N):
      for c in range(0,C):
          x_pad[i,c,:,:] = np.pad(x[i,c,:,:],pad,pad_with)
          dx_pad[i,c,:,:] = np.pad(dx[i,c,:,:],pad,pad_with)
          
  for i in range(0,N):
      for f in range(0,F):
          for m in range(0,H_o):
              hs = m*stride
              for n in range(0,W_o):
                  ws = n*stride
                  window_input = x_pad[i,:,hs:hs+HH,ws:ws+WW]
                  dw[f]+=dout[i,f,m,n]*window_input
                  db[f] +=dout[i,f,m,n]
                  dx_pad[i,:,hs:hs+HH,ws:ws+WW]+=w[f,:,:,:]*dout[i,f,m,n]
  dx = dx_pad[:,:,pad:pad+H,pad:pad+W]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N,C,H,W = x.shape
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  stride = pool_param['stride']

  H_o = 1 + (H  - ph) // stride
  W_o = 1 + (W  - pw) // stride
  out = np.zeros((N,C,H_o,W_o))

  for i in range(N):
      for c in range(C):
          for m in range(H_o):
              hs = m*stride
              for n in range(W_o):
                  ws = n*stride
                  window = x[i,c,hs:hs+ph,ws:ws+pw]
                  out[i,c,m,n] = np.max(window)
                    

    

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  x,pool_param = cache
  N,C,H,W = x.shape
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  stride = pool_param['stride']

  H_o = 1 + (H  - ph) // stride
  W_o = 1 + (W  - pw) // stride
  dx = np.zeros(x.shape)

  for i in range(N):
      for c in range(C):
          for m in range(H_o):
              hs = m*stride
              for n in range(W_o):
                  ws = n*stride
                  window = x[i,c,hs:hs+ph,ws:ws+pw]
                  max_out = np.max(window)
                  dx[i,c,hs:hs+ph,ws:ws+pw] += (window == max_out)*dout[i,c,m,n]
                    
    

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  
  N,C,H,W = x.shape
  x_t = x.transpose((0,2,3,1))
  x_t = x_t.reshape(-1,C)
  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

  x_mu = np.mean(x_t,axis = 0)
  x_var = np.var(x_t,axis = 0)

 
  running_mean = momentum * running_mean + (1 - momentum) * x_mu
  running_var = momentum * running_var + (1 - momentum) * x_var
    
  x_norm = (x_t-x_mu)/np.sqrt(x_var+eps)
  out = gamma*x_norm + beta
  cache = (x_t,x_norm,x_mu,x_var,gamma,beta,eps)

  out = out.reshape((N,H,W,C)).transpose(0,3,1,2)
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  x_t,x_norm,x_mu,x_var,gamma,beta,eps = cache
  N,C,H,W = dout.shape
  dout_t = dout.transpose((0,2,3,1))
  dout_t = dout_t.reshape(-1,C)

  N_new,_=dout_t.shape
  new_mu = x_t-x_mu

  dbeta = np.sum(dout_t,axis = 0)
  dgamma = np.sum(dout_t*x_norm,axis = 0)
  dx_norm = dout_t*gamma

  divar = np.sum(dx_norm*new_mu,axis = 0)
  dxmu1 = dx_norm/np.sqrt(x_var+eps)

  dsqrtvar = -1./(x_var+eps)*divar
  dvar = 0.5*1./np.sqrt(x_var+eps)*dsqrtvar

  dsq = 1./N_new*np.ones((N_new,C))*dvar

  dxmu2 = 2*new_mu*dsq

  dx1 = dxmu1+dxmu2
  dmu = -1*np.sum(dx1,axis=0)
  dx2 = 1./N_new*np.ones((N_new,C))*dmu

  dx = (dx1+dx2).reshape((N,H,W,C)).transpose(0,3,1,2)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta

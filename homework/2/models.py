""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class Model( object ):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__( self, nfeatures ):
        self.num_input_features = nfeatures

    def fit( self, *, X, y, A, tol ):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            A: A function of the features x, used in estimating equations.
            tol: tolerance used to exit the Newton-Raphson loop.
        """
        raise NotImplementedError()

    def predict( self, X ):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats( self, X ):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[ :, :self.num_input_features ]
        return X


class MCModel( Model ):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__( self, *, nfeatures, nclasses ):
        super().__init__( nfeatures )
        self.num_classes = nclasses


class MCLogistic( MCModel ):

    def __init__( self, *, nfeatures, nclasses ):
        super().__init__( nfeatures=nfeatures, nclasses=nclasses )
        # Define the dimensions of W (logistic) - assume 2 classes
        if isinstance( nclasses, np.ndarray ):
            self.classes = nclasses
            self.num_classes = nclasses.size
        else:
            self.classes = np.arange( nclasses )
            self.num_classes = nclasses

        self.W = np.random.randn( nfeatures + 1, 1 )  # last row is the bias term
        self.W /= 1e5 * np.abs( self.W ).max()

    def fit( self, *, X, y, A, tol ):
        # standardize array formats
        X = self._fix_test_feats( X )
        Xh = np.hstack( (X.toarray(), np.ones( (X.shape[ 0 ], 1) )) )
        y = y.reshape( -1, 1 )

        # calculate A versions
        AX = A( X )
        AXh = np.hstack( (AX.toarray(), np.ones( (AX.shape[ 0 ], 1) )) )

        # fit logistic regression
        while True:
            # conditional probability of Y_i given X and weights
            p_yi1_g_AX = self._conditional_prob_yi( AX ).reshape( -1, 1 )

            # calculate gradient and Hessian
            grad_h = np.sum( AXh * (y - p_yi1_g_AX), axis=0 )

            hess_h = np.sum( np.einsum( 'ij,ik->ijk', AXh, -AXh ) * (p_yi1_g_AX * (1 - p_yi1_g_AX)).reshape( -1, 1, 1 ),
                             axis=0 )

            # # brute force method
            # hess_h2 = np.zeros_like( hess_h )
            # for i in range( Xh.shape[ 0 ] ):
            #     xh_i = Xh[ i ]
            #     hess_h2 += np.outer( xh_i, xh_i ) * p_yi1_g_X[ i ] * (1 - p_yi1_g_X[ i ])
            #
            # # for

            # determine update and update parameters
            update = -(np.linalg.pinv( hess_h, hermitian=True ) @ grad_h)
            self.W += update.reshape( -1, 1 )  # update the parameters

            # check for update size small enough to break
            update_norm = np.linalg.norm( update )
            if update_norm <= tol:
                print()
                break

            # if

        # while

    # fit

    def predict( self, X ):
        X = self._fix_test_feats( X )
        prob_yi1 = self._conditional_prob_yi( X )
        predictions = (prob_yi1 > 0.5).astype( int ).reshape( -1, 1 )

        return predictions

    def _conditional_prob_yi( self, X ):
        """ Compute p(Y_i | X; W) """
        sig_exp = np.exp( -(X @ self.W[ :-1 ] + self.W[ -1 ]) )  # sigmoid exponential expression

        # Binary implementation
        p_yi_g_x = 1 / (1 + sig_exp)

        # # Multi-class implementation
        # p_yi_g_x = np.zeros_like( sig_exp )
        # p_yi_g_x[ :, -1 ] = 1 / (1 + sig_exp[ :, :-1 ].sum( axis=1 ))
        # p_yi_g_x[ :, :-1 ] = sig_exp[ :, :-1 ] * p_yi_g_x[ :, -1 ].reshape( -1, 1 )

        return p_yi_g_x


class MCLogisticWithL2( MCModel ):

    def __init__( self, *, nfeatures, nclasses ):
        super().__init__( nfeatures=nfeatures, nclasses=nclasses )
        # Define the dimensions of W (L2) - assume 2 classes
        if isinstance( nclasses, np.ndarray ):
            self.classes = nclasses
            self.num_classes = nclasses.size
        else:
            self.classes = np.arange( nclasses )
            self.num_classes = nclasses

        self.W = np.random.randn( nfeatures + 1, 1 )  # last row is the bias term
        self.W /= 1e5 * np.abs( self.W ).max()

    def fit( self, *, X, y, A, tol ):
        # standardize the forms
        X = self._fix_test_feats( X )
        Xh = np.hstack( (X, np.ones( (X.shape[ 0 ]), 1 )) )
        y = y.reshape( -1, 1 )

        # compute A(X)
        AX = A( X )
        AXh = np.hstack( (AX, np.ones( (X.shape[ 0 ], 1) )) )

        # TODO: fit function (L2)
        raise NotImplementedError()

    # fit

    def predict( self, X ):
        X = self._fix_test_feats( X )
        # TODO: predict function (L2)
        predictions = np.zeros( (X.shape[ 0 ], 1), dtype=np.int )

        return predictions

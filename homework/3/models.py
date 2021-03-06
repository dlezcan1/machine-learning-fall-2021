""" Model classes defined here! """

import torch
import torch.nn as nn  # added
import torch.nn.functional as F


class FeedForward( torch.nn.Module ):
    def __init__( self, hidden_dim ):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super( FeedForward, self ).__init__()
        self.fc1 = nn.Linear( 784, hidden_dim )  # 784 pixels
        self.fc2 = nn.Linear( hidden_dim, 10 )  # Logits in vectors of length 10

    def forward( self, x ):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        x = self.fc1( x )
        x = F.relu( x )
        x = self.fc2( x )

        return x


class SimpleConvNN( torch.nn.Module ):
    def __init__( self, n1_chan, n1_kern, n2_kern ):
        super( SimpleConvNN, self ).__init__()

        self.cv1 = nn.Conv2d(1, n1_chan, kernel_size = n1_kern, padding=0)
        self.cv2 = nn.Conv2d(n1_chan, 10, kernel_size = n2_kern, stride = 2, padding=0)
        
        mpsize = (28 -(n1_kern + n2_kern) + 1)//2 + 1
        self.mp = nn.MaxPool2d(mpsize)

    def forward( self, x ):
        x = x.reshape( len( x ), 1, 28, 28 )
        x = self.cv1( x )
        x = F.relu( x )
        x = self.cv2( x )
        x = F.relu( x )
        x = self.mp( x )
        x = x.view( -1, 10 )

        return x


class BestNN( torch.nn.Module ):
    # take hyperparameters from the command line args!
    def __init__( self, n1_channels, n1_kernel, n2_channels, n2_kernel, pool1,
                  n3_channels, n3_kernel, n4_channels, n4_kernel, pool2, linear_features ):
        super( BestNN, self ).__init__()
        self.cv1 = nn.Conv2d( 1, n1_channels, kernel_size=n1_kernel, padding=n1_kernel//2 )
        self.cv2 = nn.Conv2d( n1_channels, n2_channels, kernel_size=n2_kernel, padding=n2_kernel//2 )
        self.cv3 = nn.Conv2d( n2_channels, n3_channels, kernel_size=n3_kernel, padding=n3_kernel//2 )
        self.cv4 = nn.Conv2d( n3_channels, n4_channels, kernel_size=n4_kernel, padding=n4_kernel//2 )

        linear_in_size = n4_channels * ((28 // pool1) // pool2) ** 2 # number out channels times pooling dimension
        self.linear1 = nn.Linear( linear_in_size, linear_features )
        self.linear2 = nn.Linear( linear_features, 10 )

        self.pool1 = nn.MaxPool2d( pool1 )
        self.pool2 = nn.MaxPool2d( pool2 )

    def forward( self, x ):
        # first round of convolutions
        x /= 255
        x = x.view( -1, 1, 28, 28 )
        x = F.relu( self.cv1( x ) )
        x = F.relu( self.cv2( x ) )
        x = self.pool1( x )  # first max pool

        # second round of convolutions
        x = F.relu( self.cv3( x ) )
        x = F.relu( self.cv4( x ) )
        x = self.pool2( x )  # second max pool

        # linear layers
        x = x.flatten( start_dim=1 )
        x = self.linear1( x )
        x = self.linear2( x )

        return x

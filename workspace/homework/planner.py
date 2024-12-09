import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define first CNN branch
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 3x3 kernel, downsample by 2
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 3x3 kernel, maintain size
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),  # Average pooling, downsample by 2
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # 5x5 kernel, maintain size
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),  # 7x7 kernel, maintain size
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),  # Average pooling, downsample by 2
            torch.nn.Conv2d(128, 1, kernel_size=1, stride=1),  # Final layer: 1x1 kernel, produce single-channel heatmap
            torch.nn.Upsample(size=(48, 64), mode='bilinear', align_corners=False)  # Upsample to (48, 64) to match dimensionality for combined heatmap
        )# loss at 0.079 after 50 epochs but not neccessarily converged

        # Define second CNN branch
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, 2, 2),  #5x5 kernel stride 2 padding 2
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1, 1)  # Reduce to 1 output channel (heatmap)
        )#loss converges around 0.152

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 2, 1),  #3x3 kernel stride 1 padding 1
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1, 1),  # Reduce to 1 output channel (heatmap)
            #torch.nn.Upsample(size=(48, 64), mode='bilinear', align_corners=False)  # Match desired output dimensions
        )

        self.conv2b = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 3x3 kernel, downsample by 2
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 3x3 kernel, maintain size
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 1, 1)#heatmap
        )

        self.conv4 = torch.nn.Sequential(
            # Initial block: Expand channels and extract basic features
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # Downsample by 2
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            # Residual block 1
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            # Residual block 2
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample by 2
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            # Squeeze-and-Excitation Block
            torch.nn.AdaptiveAvgPool2d(1),  # Global pooling
            torch.nn.Conv2d(32, 16, kernel_size=1),  # Reduce to bottleneck
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=1),  # Expand back
            torch.nn.Sigmoid(),

            # Final heatmap generation
            torch.nn.Conv2d(32, 1, kernel_size=1, stride=1),  # Single-channel heatmap
            torch.nn.Upsample(size=(48, 64), mode='bilinear', align_corners=False)  # Match desired output dimensions
        )#around .2

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),# (B, 16, 48, 64)
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),# (B, 32, 24, 32)
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),# (B, 64, 12, 16)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),# (B, 32, 24, 32)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),# (B, 16, 48, 64)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, output_padding=1),# (B, 1, 48, 64) Heatmap layer
            #torch.nn.Upsample(size=(48, 64), mode='bilinear', align_corners=False)
        )
        #cnn outputs are (B,1,48,64) (for now)

        # # Learnable weights for combining the two heatmaps
        self.weight1 = torch.nn.Parameter(torch.tensor(0.5))  # Weight for the first branch
        self.weight2 = torch.nn.Parameter(torch.tensor(0.5))  # Weight for the second branch
        # self.weight3 = torch.nn.Parameter(torch.tensor(0))  # Weight for the third branch
        self.normalized_weight1 = 0.5
        self.normalized_weight2 = 0.5

    def forward(self, img):
        """
        Predict the aim point in image coordinates, given the supertuxkart image
        @img: (B, 3, 96, 128)
        return: (B, 2) - Predicted aim point coordinates
        """
        # Forward pass through both CNN branches
        x1 = self.conv1(img)    #Given
        #x2 = self.conv2(img)   #3x3 filters
        x2b = self.conv2b(img) #two layers of 3x3 filters
        #x3 = self.conv3(img)   #more layers kernel:3->5->7
        #x4 = self.conv4(img)   #resnet esque poor performance

        # Normalize the weights so that they sum to 1
        weight_sum = self.weight1 + self.weight2
        self.normalized_weight1 = self.weight1 / weight_sum
        self.normalized_weight2 = self.weight2 / weight_sum

        # Combine the outputs using the normalized weights
        combined_heatmap = self.normalized_weight1 * x1 + self.normalized_weight2 * x2b

        # Return the soft-argmax of the combined heatmap
        return spatial_argmax(combined_heatmap[:, 0])  # Get the first channel (since it's single-channel)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)

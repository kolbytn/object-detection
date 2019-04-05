# Define the Model

import torch
import torch.nn as nn


class CancerDetection(nn.Module):
    def __init__(self):
        super(CancerDetection, self).__init__()

        ### 1st Layer ###

        # Input is 3x512x512 Output is 64x512x512
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Input is 64x512x512 Output is 64x512x512
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Input is 64x512x512 Output is 128x256x256
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()

        ### 2nd Layer ###

        # Input is 128x256x256 Output is 128x256x256
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # Input is 128x256x256 Output is 128x256x256
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        # Input is 128x256x256 Output is 256x128x128
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu6 = nn.ReLU()

        ### 3rd Layer ###

        # Input is 256x128x128 Output is 256x128x128
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        # Input is 256x128x128 Output is 256x128x128
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()

        # Input is 256x128x128 Output is 512x64x64
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu9 = nn.ReLU()

        ### 4th Layer

        # Input is 512x64x64 Output is 512x64x64
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu10 = nn.ReLU()

        # Input is 512x64x64 Output is 512x64x64
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu11 = nn.ReLU()

        # Input is 512x64x64 Output is 1024x32x32
        self.conv12 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.relu12 = nn.ReLU()

        ### 5th Layer ###

        # Input is 1024x32x32 Output is 1024x32x32
        self.conv13 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.relu13 = nn.ReLU()

        # Input is 1024x32x32 Output is 1024x32x32
        self.conv14 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.relu14 = nn.ReLU()

        # Input is 1024x32x32 Output is 512x64x64
        self.conv15 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.relu15 = nn.ReLU()

        ### 4th Layer (Upsampled) ###

        # 4th Layer Concat, Output is 1024x64,64
        # self.concat4 = torch.cat(self.relu11, self.relu15)

        # Input is 1024x64x64 Output is 512x64x64
        self.conv16 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.relu16 = nn.ReLU()
        self
        # Input is 512x64x64 Output is 512x64x64
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu17 = nn.ReLU()

        # Input is 512x64x64 Output is 256x128x128
        self.conv18 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.relu18 = nn.ReLU()

        ### 3rd Layer (Upsampled) ###

        # 3rd Layer Concat, Output is 512x128x128
        # self.concat3 = torch.cat(self.relu8, self.relu18)

        # Input is 512x128x128 Output is 256x128x128
        self.conv19 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.relu19 = nn.ReLU()

        # Input is 256x128x128 Output is 256x128x128
        self.conv20 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu20 = nn.ReLU()

        # Input is 256x128x128 Output is 128x256x256
        self.conv21 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.relu21 = nn.ReLU()

        ### 2nd Layer (Upsampled) ###

        # 4th Layer Concat, Output is 256x256x256
        # self.concat2 = torch.cat(self.relu5, self.relu21)

        # Input is 256x256x256 Output is 128x256x256
        self.conv22 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu22 = nn.ReLU()

        # Input is 128x256x256 Output is 128x256x256
        self.conv23 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu23 = nn.ReLU()

        # Input is 128x256x256 Output is 64x512x512
        self.conv24 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.relu24 = nn.ReLU()

        ### 1st Layer (Upsampled) ###

        # 4th Layer Concat, Output is 128x512x512
        # self.concat1 = torch.cat(self.relu2, self.relu24)

        # Input is 128x512x512 Output is 64x512x512
        self.conv25 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu25 = nn.ReLU()

        # Input is 64x512x512 Output is 64x512x512
        self.conv26 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu26 = nn.ReLU()

        # Input is 64x512x512 Output is 2x512x512
        self.conv27 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        # No activation on final layer

    def forward(self, input):
        # Layer 1
        conv1_out = self.conv1(input)
        relu1_out = self.relu1(conv1_out)
        conv2_out = self.conv2(relu1_out)
        relu2_out = self.relu2(conv2_out)
        conv3_out = self.conv3(relu2_out)
        relu3_out = self.relu3(conv3_out)

        # Layer 2
        conv4_out = self.conv4(relu3_out)
        relu4_out = self.relu4(conv4_out)
        conv5_out = self.conv5(relu4_out)
        relu5_out = self.relu5(conv5_out)
        conv6_out = self.conv6(relu5_out)
        relu6_out = self.relu6(conv6_out)

        # Layer 3
        conv7_out = self.conv7(relu6_out)
        relu7_out = self.relu7(conv7_out)
        conv8_out = self.conv8(relu7_out)
        relu8_out = self.relu8(conv8_out)
        conv9_out = self.conv9(relu8_out)
        relu9_out = self.relu9(conv9_out)

        # Layer 4
        conv10_out = self.conv10(relu9_out)
        relu10_out = self.relu10(conv10_out)
        conv11_out = self.conv11(relu10_out)
        relu11_out = self.relu11(conv11_out)
        conv12_out = self.conv12(relu11_out)
        relu12_out = self.relu12(conv12_out)

        # Layer 5
        conv13_out = self.conv13(relu12_out)
        relu13_out = self.relu13(conv13_out)
        conv14_out = self.conv14(relu13_out)
        relu14_out = self.relu14(conv14_out)
        conv15_out = self.conv15(relu14_out)
        relu15_out = self.relu15(conv15_out)

        # Layer 4 (Upsampled)

        # 4th Layer Concat
        concat4_out = torch.cat((relu11_out, relu15_out), dim=1)

        conv16_out = self.conv16(concat4_out)
        relu16_out = self.relu16(conv16_out)
        conv17_out = self.conv17(relu16_out)
        relu17_out = self.relu17(conv17_out)
        conv18_out = self.conv18(relu17_out)
        relu18_out = self.relu18(conv18_out)

        # 3th Layer Concat
        concat3_out = torch.cat((relu8_out, relu18_out), dim=1)

        # Layer 3 (Upsampled)
        conv19_out = self.conv19(concat3_out)
        relu19_out = self.relu19(conv19_out)
        conv20_out = self.conv20(relu19_out)
        relu20_out = self.relu20(conv20_out)
        conv21_out = self.conv21(relu20_out)
        relu21_out = self.relu21(conv21_out)

        # Layer 2 (Upsampled)

        # 2nd Layer Concat
        concat2_out = torch.cat((relu5_out, relu21_out), dim=1)

        conv22_out = self.conv22(concat2_out)
        relu22_out = self.relu22(conv22_out)
        conv23_out = self.conv23(relu22_out)
        relu23_out = self.relu23(conv23_out)
        conv24_out = self.conv24(relu23_out)
        relu24_out = self.relu24(conv24_out)

        # Layer 1 (Upsampled)

        # 4th Layer Concat
        concat1_out = torch.cat((relu2_out, relu24_out), dim=1)

        conv25_out = self.conv25(concat1_out)
        relu25_out = self.relu25(conv25_out)
        conv26_out = self.conv26(relu25_out)
        relu26_out = self.relu26(conv26_out)
        output = self.conv27(relu26_out)

        return output

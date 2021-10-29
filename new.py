# Imports
from pathlib import Path
import blobconverter
import numpy as np
import math
import cv2
import depthai as dai
import re
import time

# Set BlobConverter Information
openvinoVersion = "2021.4"
p = dai.Pipeline()
p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

# Download Model for use
size = (300,300)
# size = (544,320)
# nnPath = blobconverter.from_zoo("person-detection-retail-0013", shaves = 6)
nnPath = blobconverter.from_zoo("face-detection-retail-0004", shaves=6)
# Labels
labelMap = ["background", "person"]

# Set Resolution of BW Cameras
bw_resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# Set Resolution of Color Camera
color_resolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P

# Create RGB Camera Node
rgb_cam = p.create(dai.node.ColorCamera)
rgb_cam.setPreviewSize(size[0], size[1])
rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb_cam.setResolution(color_resolution)
rgb_cam.setInterleaved(False)
rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
rgb_cam.setPreviewKeepAspectRatio(False)

# Create L-MONO Node
l_cam = p.create(dai.node.MonoCamera)
l_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
l_cam.setResolution(bw_resolution)

# Create R-MONO Node
r_cam = p.create(dai.node.MonoCamera)
r_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
r_cam.setResolution(bw_resolution)

# Create Depth Node
stereo = p.create(dai.node.StereoDepth)
stereo.setLeftRightCheck(False)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)
stereo.initialConfig.setConfidenceThreshold(255)
# Link R-Mono to Stereo
r_cam.out.link(stereo.right)

# Link L-Mono to Stero
l_cam.out.link(stereo.left)

# Create NN Node
nn = p.create(dai.node.MobileNetSpatialDetectionNetwork)
nn.setBlobPath(str(Path(nnPath).resolve().absolute()))
# ignore detections below 50%
nn.setConfidenceThreshold(0.5)
nn.input.setBlocking(True)

# Link RGB_Cam to NN
rgb_cam.preview.link(nn.input)

# Link Stereo to NN
stereo.depth.link(nn.inputDepth)

# Create Blurring Node
blur = p.create(dai.node.NeuralNetwork)
blur.setBlobPath(str(Path(__file__).parent/'out'/'model.blob'))
nn.passthrough.link(blur.input)

# Create RGB Out Node
xout_rgb = p.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# Link NN.RGB Passthrough -> XLinkOut
nn.passthrough.link(xout_rgb.input)
# rgb_cam.video.link(xout_rgb.input)
# blur.out.link(xout_rgb.input)

# Create Depth-Output Node
xout_depth = p.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")

# Link NN.passthroughDepth -> XLinkOut
# stereo.depth.link(xout_depth.input)
nn.passthroughDepth.link(xout_depth.input)

# Create NN-Out Node
xout_nn = p.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

# Link NN.Out -> NNout (Information about detection)
nn.out.link(xout_nn.input)

# # Create BoundingBox Output Node
xout_bb = p.create(dai.node.XLinkOut)
xout_bb.setStreamName("bb")

# # Link NN.boundingBoxMapping -> bb_out
nn.boundingBoxMapping.link(xout_bb.input)

cv2.namedWindow("Depth")
cv2.namedWindow("Image")
cv2.moveWindow("Depth", 0,0)
# cv2.moveWindow("Image", 912,35)
cv2.moveWindow("Image", 0, 513)
# Connect to Device and Start Pipeline
with dai.Device(p) as dev:
    rgbQueue = dev.getOutputQueue(name = "rgb", maxSize = 10, blocking = False)
    depthQueue = dev.getOutputQueue(name="depth", maxSize= 10, blocking=False)
    nnQueue = dev.getOutputQueue(name="nn", maxSize=10, blocking=False)
    bbQueue = dev.getOutputQueue(name="bb", maxSize=10, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while True:
        rgb_in = rgbQueue.get()
        depth_in = depthQueue.get()
        nn_out = nnQueue.get()
        
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter/(current_time - startTime)
            counter = 0
            startTime =  current_time
        # rgbFrame = rgb_in.getFirstLayerFp16()
        # rgbFrame = np.array(rgbFrame, dtype=np.uint8)
        # shape = (300, 300, 3)
        # rgbFrame = rgbFrame.reshape(shape)
        rgbFrame = rgb_in.getCvFrame()
        rgbFrame = cv2.resize(rgbFrame, (int(1600*0.45), int(900*0.45)))
        
        depthFrame = depth_in.getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_TURBO)
        depthFrameColor = cv2.resize(depthFrameColor, (int(1600*0.45), int(900*0.45)))

        detections = nn_out.detections
        if len(detections) != 0:
            bbMapping = bbQueue.get()
            roiDatas = bbMapping.getConfigData()

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmid = int((int(topLeft.x)-int(bottomRight.x))/2)
                xmin = int(topLeft.x)+xmid
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)+xmid
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), 255, 10)
        rgb_height = rgbFrame.shape[0]
        rgb_width = rgbFrame.shape[1]

        for detection in detections:
            rgb_x1 = int(detection.xmin*rgb_width)
            rgb_x2 = int(detection.xmax*rgb_width)
            rgb_y1 = int(detection.ymin*rgb_height)
            rgb_y2 = int(detection.ymax*rgb_height)

            cv2.rectangle(rgbFrame, (rgb_x1, rgb_y1), (rgb_x2, rgb_y2), 255, 10)
            cv2.putText(rgbFrame, f"Z: {int(detection.spatialCoordinates.z)/1000} m",
                        (rgb_x1+int((rgb_x2-rgb_x1)/4), rgb_y1+int((rgb_y2-rgb_y1)/2)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        cv2.putText(rgbFrame, "NN fps: {:.2f}".format(fps), (2, rgbFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
        cv2.putText(rgbFrame, "COLOR IMAGE", (int(rgbFrame.shape[0]/2)+50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(depthFrameColor, "DEPTH IMAGE", (int(
            depthFrameColor.shape[0]/2)+50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 0), 2)

        cv2.imshow("Depth", depthFrameColor)
        cv2.imshow("Image", rgbFrame)

        if cv2.waitKey(1) == ord('q'):
            break

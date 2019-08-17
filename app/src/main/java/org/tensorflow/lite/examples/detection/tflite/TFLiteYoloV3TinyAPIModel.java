/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.app.Activity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteYoloV3TinyAPIModel extends YoloClassifier {
  // outputLocations
  private float[][][][] output1 = new float[1][8][8][18];
  private float[][][][] output2 = new float[1][16][16][18];

  // anchors
  private int[] anchors1 = new int[] { 39,8, 44,13, 71,19 };
  private int[] anchors2 = new int[] { 18,6, 25,7, 26,11 };

  private final YoloOutputs outputs;

  public TFLiteYoloV3TinyAPIModel(Activity activity, Device device, int numThreads, float minimumConfidence) throws IOException {
    super(activity, device, numThreads, minimumConfidence);

    outputs = new YoloOutputs()
            .addOutput(8, 8, 3, getNumLabels(), new int[] { 39,8, 44,13, 71,19 })
            .addOutput(16, 16, 3, getNumLabels(), new int[] { 18,6, 25,7, 26,11 });
  }

  @Override
  protected void runInference() {
    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
//    outputMap.put(0, output1);
//    outputMap.put(1, output2);

    for (YoloOutput yoloOutput : outputs.getOutputs()) {
      outputMap.put(yoloOutput.getIndex(), yoloOutput.getOutput());
    }

    tflite.runForMultipleInputsOutputs(inputArray, outputMap);
  }

  @Override
  protected List<Recognition> postprocessResults() {
    final List<Recognition> recognitions = new ArrayList<>();

//    recognitions.addAll(decodeNetout(output1[0], 8, 8, anchors1, minimumConfidence));
//    recognitions.addAll(decodeNetout(output2[0], 16, 16, anchors2, minimumConfidence));

    for (YoloOutput yoloOutput : outputs.getOutputs()) {
      recognitions.addAll(decodeNetout(yoloOutput.getOutput()[0], yoloOutput.getGridHeight(), yoloOutput.getGridWidth(), yoloOutput.getAnchors(), minimumConfidence));
    }

    correctYoloBoxes(recognitions);

    final List<Recognition> filteredRecognitions = doNms(recognitions);

    return filteredRecognitions;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.putFloat(((pixelValue >> 16) & 0xFF) / IMAGE_MAX_VALUE);
    imgData.putFloat(((pixelValue >> 8) & 0xFF) / IMAGE_MAX_VALUE);
    imgData.putFloat((pixelValue & 0xFF) / IMAGE_MAX_VALUE);
  }

  @Override
  public int getImageSizeX() {
    return 256;
  }

  @Override
  public int getImageSizeY() {
    return 256;
  }

  @Override
  protected int getNumBytesPerChannel() {
    return 4;
  }

  @Override
  protected String getLabelPath() {
    return "mylabelmap.txt";
  }

  @Override
  protected String getModelPath() {
    return "vC1_model_best_weights.tflite";
  }
}
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
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteYoloV3TinyDebugAPIModel extends YoloClassifier {
  // outputLocations
  private float[][][][] output1 = new float[1][8][8][18];
  private float[][][][] output2 = new float[1][16][16][18];

  // anchors
  private int[] anchors1 = new int[] { 39,8, 44,13, 71,19 };
  private int[] anchors2 = new int[] { 18,6, 25,7, 26,11 };

  private final YoloOutputs outputs;

  private final int[][][] loadedIntValues;

  public TFLiteYoloV3TinyDebugAPIModel(Activity activity, Device device, int numThreads, float minimumConfidence) throws IOException {
    super(activity, device, numThreads, minimumConfidence);

    outputs = new YoloOutputs()
            .addOutput(8, 8, 3, getNumLabels(), new int[] { 39,8, 44,13, 71,19 })
            .addOutput(16, 16, 3, getNumLabels(), new int[] { 18,6, 25,7, 26,11 });

    loadedIntValues = new int[getImageSizeX()][getImageSizeY()][DIM_CHANNEL_SIZE];

    try (BufferedReader br = new BufferedReader(new InputStreamReader(activity.getAssets().open("g0010119_channel_0.txt")))) {
      int lineNumber = 0;
      String line;

      while ((line = br.readLine()) != null) {
        String[] columns = line.split(" ");

        for (int c = 0; c < columns.length; c++) {
          loadedIntValues[lineNumber][c][0] = Integer.valueOf(columns[c]);
        }

        lineNumber++;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    try (BufferedReader br = new BufferedReader(new InputStreamReader(activity.getAssets().open("g0010119_channel_1.txt")))) {
      int lineNumber = 0;
      String line;

      while ((line = br.readLine()) != null) {
        String[] columns = line.split(" ");

        for (int c = 0; c < columns.length; c++) {
          loadedIntValues[lineNumber][c][1] = Integer.valueOf(columns[c]);
        }

        lineNumber++;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    try (BufferedReader br = new BufferedReader(new InputStreamReader(activity.getAssets().open("g0010119_channel_2.txt")))) {
      int lineNumber = 0;
      String line;

      while ((line = br.readLine()) != null) {
        String[] columns = line.split(" ");

        for (int c = 0; c < columns.length; c++) {
          loadedIntValues[lineNumber][c][2] = Integer.valueOf(columns[c]);
        }

        lineNumber++;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  protected void runInference() {
    imgData.rewind();

    for (int i = 0; i < getImageSizeX(); ++i) {
      for (int j = 0; j < getImageSizeY(); ++j) {
        imgData.putFloat(loadedIntValues[i][j][2] / IMAGE_MAX_VALUE);
        imgData.putFloat(loadedIntValues[i][j][1] / IMAGE_MAX_VALUE);
        imgData.putFloat(loadedIntValues[i][j][0] / IMAGE_MAX_VALUE);
      }
    }

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

//  public List<Recognition> recognizeImage(final Bitmap bitmap, final int[][][] loadedIntValues) {
//    // Log this method so that it can be analyzed with systrace.
//    Trace.beginSection("recognizeImage");
//
//    Trace.beginSection("preprocessBitmap");
//    // Preprocess the image data from 0-255 int to normalized float based
//    // on the provided parameters.
//    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//
//    imgData.rewind();
//    // Original
//    for (int i = 0; i < inputSize; ++i) {
//      for (int j = 0; j < inputSize; ++j) {
//        int pixelValue = intValues[i * inputSize + j];
//        if (isModelQuantized) {
//          // Quantized model
//          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
//          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
//          imgData.put((byte) (pixelValue & 0xFF));
//        } else { // Float model
//          imgData.putFloat(((pixelValue >> 16) & 0xFF) / IMAGE_MAX_VALUE);
//          imgData.putFloat(((pixelValue >> 8) & 0xFF) / IMAGE_MAX_VALUE);
//          imgData.putFloat((pixelValue & 0xFF) / IMAGE_MAX_VALUE);
//        }
//      }
//    }
//    // Test 1 - OK
////    for (int i = 0; i < inputSize; ++i) {
////      for (int j = 0; j < inputSize; ++j) {
////        if (isModelQuantized) {
////          // Quantized model
////          imgData.put((byte) loadedIntValues[i][j][2]);
////          imgData.put((byte) loadedIntValues[i][j][1]);
////          imgData.put((byte) loadedIntValues[i][j][0]);
////        } else { // Float model
////          imgData.putFloat(loadedIntValues[i][j][2] / IMAGE_MAX_VALUE);
////          imgData.putFloat(loadedIntValues[i][j][1] / IMAGE_MAX_VALUE);
////          imgData.putFloat(loadedIntValues[i][j][0] / IMAGE_MAX_VALUE);
////        }
////      }
////    }
//    // Test 2
////    float[][][][] floatValues = new float[1][inputSize][inputSize][3];
////
////    for (int y = 0; y < inputSize; y++) {
////      for (int x = 0; x < inputSize; x++) {
////        floatValues[0][y][x][0] = loadedIntValues[y][x][2] / 255.0f;
////        floatValues[0][y][x][1] = loadedIntValues[y][x][1] / 255.0f;
////        floatValues[0][y][x][2] = loadedIntValues[y][x][0] / 255.0f;
////      }
////    }
//    // Test 3
////    float[][][][] floatValues = new float[1][inputSize][inputSize][3];
////
////    for (int y = 0; y < inputSize; y++) {
////      for (int x = 0; x < inputSize; x++) {
////        floatValues[0][y][x][0] = 0.11253756f;
////        floatValues[0][y][x][1] = 0.11253756f;
////        floatValues[0][y][x][2] = 0.11253756f;
////      }
////    }
//
////    floatValues[0][0][0][0] = 1.0f;
////    floatValues[0][0][0][1] = 1.0f;
////    floatValues[0][0][0][2] = 1.0f;
//
//    Trace.endSection(); // preprocessBitmap
//
//    // Copy the input data into TensorFlow.
//    Trace.beginSection("feed");
//    output1 = new float[1][8][8][18];
//    output2 = new float[1][16][16][18];
//
//    ByteBuffer outputBB1 = ByteBuffer.allocateDirect(1 * 8 * 8 * 18 * 4);
//    outputBB1.order(ByteOrder.nativeOrder());
//    outputBB1.rewind();
//
//    ByteBuffer outputBB2 = ByteBuffer.allocateDirect(1 * 16 * 16 * 18 * 4);
//    outputBB1.order(ByteOrder.nativeOrder());
//    outputBB2.rewind();
//
//    Object[] inputArray = {imgData};
////    Object[] inputArray = {floatValues};
//    Map<Integer, Object> outputMap = new HashMap<>();
//    outputMap.put(0, output1);
//    outputMap.put(1, output2);
////    outputMap.put(0, outputBB1);
////    outputMap.put(1, outputBB2);
//    Trace.endSection();
//
//    // Run the inference call.
//    Trace.beginSection("run");
//    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
//    Trace.endSection();
//
//    // Show the best detections.
//    // after scaling them back to the input size.
//    final List<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
//
//    recognitions.addAll(decodeNetout(output1[0], 8, 8, anchors1, 256, 256));
//    recognitions.addAll(decodeNetout(output2[0], 16, 16, anchors2, 256, 256));
//
////    correctYoloBoxes(recognitions, 256, 256, bitmap.getWidth(), bitmap.getHeight());
//    correctYoloBoxes(recognitions, 256, 256, 640, 480);
//
//    final List<Recognition> filteredRecognitions = doNms(recognitions, .2f);
//
//    Trace.endSection(); // "recognizeImage"
//    return filteredRecognitions;
//  }
}

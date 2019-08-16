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
public class TFLiteYoloV3TinyDebugAPIModel implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MAX_VALUE = 255.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations
  private float[][][][] output1;
  private float[][][][] output2;
  // anchors
  private int[] anchors1 = new int[] { 39,8, 44,13, 71,19 };
  private int[] anchors2 = new int[] { 18,6, 25,7, 26,11 };

  private ByteBuffer imgData;

  private Interpreter tfLite;

  private TFLiteYoloV3TinyDebugAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Classifier create(final AssetManager assetManager, final String modelFilename, final String labelFilename, final int inputSize, final boolean isQuantized) throws IOException {
    final TFLiteYoloV3TinyDebugAPIModel d = new TFLiteYoloV3TinyDebugAPIModel();

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);

    try(BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput))) {
      String line;
      while ((line = br.readLine()) != null) {
        LOGGER.w(line);
        d.labels.add(line);
      }
    }

    d.inputSize = inputSize;

    try {
      Interpreter.Options interpreterOptions = new Interpreter.Options().setNumThreads(NUM_THREADS).setAllowFp16PrecisionForFp32(Boolean.TRUE);

      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), interpreterOptions);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;

    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }

    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.output1 = new float[1][8][8][18];
    d.output2 = new float[1][16][16][18];

    return d;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    throw new RuntimeException("Not implemented");
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap, final float minimumConfidence) {
    throw new RuntimeException("Not implemented");
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap, final int[][][] loadedIntValues) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    // Original
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat(((pixelValue >> 16) & 0xFF) / IMAGE_MAX_VALUE);
          imgData.putFloat(((pixelValue >> 8) & 0xFF) / IMAGE_MAX_VALUE);
          imgData.putFloat((pixelValue & 0xFF) / IMAGE_MAX_VALUE);
        }
      }
    }
    // Test 1 - OK
//    for (int i = 0; i < inputSize; ++i) {
//      for (int j = 0; j < inputSize; ++j) {
//        if (isModelQuantized) {
//          // Quantized model
//          imgData.put((byte) loadedIntValues[i][j][2]);
//          imgData.put((byte) loadedIntValues[i][j][1]);
//          imgData.put((byte) loadedIntValues[i][j][0]);
//        } else { // Float model
//          imgData.putFloat(loadedIntValues[i][j][2] / IMAGE_MAX_VALUE);
//          imgData.putFloat(loadedIntValues[i][j][1] / IMAGE_MAX_VALUE);
//          imgData.putFloat(loadedIntValues[i][j][0] / IMAGE_MAX_VALUE);
//        }
//      }
//    }
    // Test 2
//    float[][][][] floatValues = new float[1][inputSize][inputSize][3];
//
//    for (int y = 0; y < inputSize; y++) {
//      for (int x = 0; x < inputSize; x++) {
//        floatValues[0][y][x][0] = loadedIntValues[y][x][2] / 255.0f;
//        floatValues[0][y][x][1] = loadedIntValues[y][x][1] / 255.0f;
//        floatValues[0][y][x][2] = loadedIntValues[y][x][0] / 255.0f;
//      }
//    }
    // Test 3
//    float[][][][] floatValues = new float[1][inputSize][inputSize][3];
//
//    for (int y = 0; y < inputSize; y++) {
//      for (int x = 0; x < inputSize; x++) {
//        floatValues[0][y][x][0] = 0.11253756f;
//        floatValues[0][y][x][1] = 0.11253756f;
//        floatValues[0][y][x][2] = 0.11253756f;
//      }
//    }

//    floatValues[0][0][0][0] = 1.0f;
//    floatValues[0][0][0][1] = 1.0f;
//    floatValues[0][0][0][2] = 1.0f;

    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    output1 = new float[1][8][8][18];
    output2 = new float[1][16][16][18];

    ByteBuffer outputBB1 = ByteBuffer.allocateDirect(1 * 8 * 8 * 18 * 4);
    outputBB1.order(ByteOrder.nativeOrder());
    outputBB1.rewind();

    ByteBuffer outputBB2 = ByteBuffer.allocateDirect(1 * 16 * 16 * 18 * 4);
    outputBB1.order(ByteOrder.nativeOrder());
    outputBB2.rewind();

    Object[] inputArray = {imgData};
//    Object[] inputArray = {floatValues};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, output1);
    outputMap.put(1, output2);
//    outputMap.put(0, outputBB1);
//    outputMap.put(1, outputBB2);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

    // Show the best detections.
    // after scaling them back to the input size.
    final List<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);

    recognitions.addAll(decodeNetout(output1[0], 8, 8, anchors1, 256, 256));
    recognitions.addAll(decodeNetout(output2[0], 16, 16, anchors2, 256, 256));

//    correctYoloBoxes(recognitions, 256, 256, bitmap.getWidth(), bitmap.getHeight());
    correctYoloBoxes(recognitions, 256, 256, 640, 480);

    final List<Recognition> filteredRecognitions = doNms(recognitions, .2f);

    Trace.endSection(); // "recognizeImage"
    return filteredRecognitions;
  }

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;

    for (final float val : vals) {
      max = Math.max(max, val);
    }

    float sum = 0.0f;

    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }

    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  private List<Recognition> decodeNetout(float[][][] netout, int gridHeight, int gridWidth, int[] anchors, int netHeight, int netWidth) {
    int NUM_BOXES_PER_BLOCK = 3;
    int NUM_CLASSES = 1;
    float THRESHOLD = 0.5f;

    // Find the best detections.
    List<Recognition> detections = new ArrayList<>();

    for (int y = 0; y < gridHeight; ++y) {
      for (int x = 0; x < gridWidth; ++x) {
        for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
          final int offset = (NUM_CLASSES + 5) * b;

          final float confidence = expit(netout[y][x][offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = netout[y][x][offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass >  THRESHOLD) {
            final float xPos = (x + expit(netout[y][x][offset + 0])) / gridWidth;
            final float yPos = (y + expit(netout[y][x][offset + 1])) / gridHeight;

            final float w = (float) (Math.exp(netout[y][x][offset + 2]) * anchors[2 * b + 0]) / netWidth;
            final float h = (float) (Math.exp(netout[y][x][offset + 3]) * anchors[2 * b + 1]) / netHeight;

            final RectF rect = new RectF(
                    Math.max(0, xPos - w / 2),
                    Math.max(0, yPos - h / 2),
                    Math.min(netWidth - 1, xPos + w / 2),
                    Math.min(netHeight - 1, yPos + h / 2));

            if (confidenceInClass > 0.01) {
              detections.add(new Recognition("" + offset, labels.get(detectedClass), confidenceInClass, rect));
            }
          }
        }
      }
    }

    return detections;
  }

  private void correctYoloBoxes(List<Recognition> recognitions, int netWidth, int netHeight, int imageWidth, int imageHeight) {
    int new_w;
    int new_h;

    if (((float) netWidth / imageWidth) < ((float) netHeight / imageHeight)) {
      new_w = netWidth;
      new_h = (imageHeight * netWidth) / imageWidth;
    } else {
      new_h = netWidth;
      new_w = (imageWidth * netHeight) / imageHeight;
    }

    for (Recognition recognition : recognitions) {
      float x_offset = (netWidth - new_w) / 2.f / netWidth;
      float x_scale = (float) new_w / netWidth;

      float y_offset = (netHeight - new_h) / 2.f / netHeight;
      float y_scale = (float) new_h / netHeight;

      RectF correctedLocation = new RectF((int) ((recognition.getLocation().left - x_offset) / x_scale * imageWidth),
                (int) ((recognition.getLocation().top - y_offset) / y_scale * imageHeight),
                (int) ((recognition.getLocation().right - x_offset) / x_scale * imageWidth),
                (int) ((recognition.getLocation().bottom - y_offset) / y_scale * imageHeight));

      recognition.setLocation(correctedLocation);
    }
  }

  private List<Recognition> doNms(List<Recognition> recognitions, float nmsThreashold) {
    if (recognitions.isEmpty()) {
      return new ArrayList<>();
    }

    final PriorityQueue<Recognition> priorityQueue = new PriorityQueue<>(recognitions.size(), new Comparator<Recognition>() {
      @Override
      public int compare(final Recognition lhs, final Recognition rhs) {
        // Intentionally reversed to put high confidence at the head of the queue.
        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
      }
    });

    priorityQueue.addAll(recognitions);

    final List<Recognition> predictions = new ArrayList<>();
    predictions.add(priorityQueue.poll()); // best prediction

    Recognition currentPrediction = null;
    while ((currentPrediction = priorityQueue.poll()) != null) {
      boolean overlaps = false;

      for (Recognition previousPrediction : predictions) {
        float intersectProportion = 0f;

        RectF primary = previousPrediction.getLocation();
        RectF secondary =  currentPrediction.getLocation();

        if (primary.left < secondary.right && primary.right > secondary.left && primary.top < secondary.bottom && primary.bottom > secondary.top) {
          float intersection = Math.max(0, Math.min(primary.right, secondary.right) - Math.max(primary.left, secondary.left)) *
                  Math.max(0, Math.min(primary.bottom, secondary.bottom) - Math.max(primary.top, secondary.top));

          float main = Math.abs(primary.right - primary.left) * Math.abs(primary.bottom - primary.top);

          intersectProportion = intersection / main;
        }

        overlaps = overlaps || (intersectProportion > nmsThreashold);
      }

      if (!overlaps) {
        predictions.add(currentPrediction);
      }
    }

    return predictions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }
}

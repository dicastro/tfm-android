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
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/** Generic interface for interacting with different recognition engines. */
public abstract class Classifier {
  private static final Logger LOGGER = new Logger();

  /** The model type used for classification. */
  public enum Model {
    V3_TINY_256, V3_TINY_416, V3_TINY_256_DEBUG
  }

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU, GPU
  }

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  /** Preallocated buffers for storing image data in. */
  private int[] intValues = new int[getImageSizeX() * getImageSizeY()];

  /** Options for configuring the Interpreter. */
//  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  private final int numThreads;
  private final Device device;

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel;

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;

  /** Optional GPU delegate for accleration. */
  private GpuDelegate gpuDelegate;

  /** Number of channels of inputs */
  protected static final int DIM_CHANNEL_SIZE = 3;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private Interpreter tflite;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  protected ByteBuffer imgData;

  protected float minimumConfidence;

  private long tfliteThread;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param model The model to use for classification.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, Model model, Device device, int numThreads, float minimumConfidence) throws IOException {
    Classifier classifier;

    switch (model) {
      case V3_TINY_256_DEBUG:
        classifier = new TFLiteYoloV3Tiny256DebugAPIModel(activity, device, numThreads, minimumConfidence);
        break;
      case V3_TINY_416:
        classifier = new TFLiteYoloV3Tiny416APIModel(activity, device, numThreads, minimumConfidence);
        break;
      default:
        classifier = new TFLiteYoloV3Tiny256APIModel(activity, device, numThreads, minimumConfidence);
        break;
    }

    return classifier;
  }

  /** Initializes a {@code Classifier}. */
  protected Classifier(Activity activity, Device device, int numThreads, float minimumConfidence) throws IOException {
    tfliteModel = loadModelFile(activity);

    this.device = device;
    this.numThreads = numThreads;

    this.minimumConfidence = minimumConfidence;

    labels = loadLabelList(activity);
    imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * getImageSizeX() * getImageSizeY() * DIM_CHANNEL_SIZE * getNumBytesPerChannel());
    imgData.order(ByteOrder.nativeOrder());

    LOGGER.d("Created a Tensorflow Lite Image Classifier.");
  }

  protected Interpreter getInterpreter() {
    if (tflite == null || tfliteThread != Thread.currentThread().getId()) {
      tfliteThread = Thread.currentThread().getId();

      final Interpreter.Options tfliteOptions = new Interpreter.Options();

      switch (device) {
        case GPU:
          gpuDelegate = new GpuDelegate();
          tfliteOptions.addDelegate(gpuDelegate);
          break;
        case CPU:
          break;
      }

      tfliteOptions.setNumThreads(numThreads);

      tflite = new Interpreter(tfliteModel, tfliteOptions);
    }

    return tflite;
  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity) throws IOException {
    List<String> labels = new ArrayList<String>();

    try (BufferedReader reader = new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())))) {
      String line;
      while ((line = reader.readLine()) != null) {
        labels.add(line);
      }
    }

    return labels;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Closes the interpreter and model to release resources. */
  public void close() {
    if (tflite != null) {
      tflite.close();
      tflite = null;
    }

    if (gpuDelegate != null) {
      gpuDelegate.close();
      gpuDelegate = null;
    }

    tfliteModel = null;
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }

    imgData.rewind();

    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < getImageSizeX(); ++i) {
      for (int j = 0; j < getImageSizeY(); ++j) {
        final int pixelValue = intValues[pixel++];
        addPixelValue(pixelValue);
      }
    }
    long endTime = SystemClock.uptimeMillis();
    LOGGER.v("Timecost to put values into ByteBuffer: " + (endTime - startTime));
  }

  /** Runs inference and returns the classification results. */
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    convertBitmapToByteBuffer(bitmap);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("runInference");
    long startTime = SystemClock.uptimeMillis();
    runInference();
    long endTime = SystemClock.uptimeMillis();
    Trace.endSection();
    LOGGER.v("Timecost to run model inference: " + (endTime - startTime));

    final List<Recognition> recognitions = postprocessResults();

    Trace.endSection();
    return recognitions;
  }

  /**
   * Get the image size along the x axis.
   *
   * @return
   */
  public abstract int getImageSizeX();

  /**
   * Get the image size along the y axis.
   *
   * @return
   */
  public abstract int getImageSizeY();

  /**
   * Get the number of bytes that is used to store a single color channel value.
   *
   * @return
   */
  protected abstract int getNumBytesPerChannel();

  /**
   * Get the name of the label file stored in Assets.
   *
   * @return
   */
  protected abstract String getLabelPath();

  /**
   * Get the name of the model file stored in Assets.
   *
   * @return
   */
  protected abstract String getModelPath();

  /**
   * Add pixelValue to byteBuffer.
   *
   * @param pixelValue
   */
  protected abstract void addPixelValue(int pixelValue);

  /**
   * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
   * provided by getProbability().
   *
   * <p>This additional method is necessary, because we don't have a common base for different
   * primitive data types.
   */
  protected abstract void runInference();

  protected abstract List<Recognition> postprocessResults();

  /**
   * Get the total number of labels.
   *
   * @return
   */
  protected int getNumLabels() {
    return labels.size();
  }

  protected String getLabel(int pos) {
    return labels.get(pos);
  }

  protected class YoloOutput {
    private final int index;
    private final int gridWidth;
    private final int gridHeight;
    private final int numBoxesPerBlock;
    private float [][][][] output;
    private final int [] anchors;

    public YoloOutput(int index, int gridWidth, int gridHeight, int numBoxesPerBlock, int numClasses, int[] anchors) {
      this.index = index;
      this.gridWidth = gridWidth;
      this.gridHeight = gridHeight;
      this.numBoxesPerBlock = numBoxesPerBlock;
      this.anchors = anchors;

      output = new float[1][gridWidth][gridHeight][numBoxesPerBlock * (5 + numClasses)];
    }

    public int getIndex() {
      return index;
    }

    public int getGridWidth() {
      return gridWidth;
    }

    public int getGridHeight() {
      return gridHeight;
    }

    public int getNumBoxesPerBlock() {
      return numBoxesPerBlock;
    }

    public int[] getAnchors() {
      return anchors;
    }

    public float[][][][] getOutput() {
      return output;
    }
  }

  public class YoloOutputs {
    private List<YoloOutput> outputs = new ArrayList<>();

    public YoloOutputs addOutput(int gridWidth, int gridHeight, int numBoxesPerBlock, int numClasses, int[] anchors) {
      outputs.add(new YoloOutput(outputs.size(), gridWidth, gridHeight, numBoxesPerBlock, numClasses, anchors));
      return this;
    }

    public List<YoloOutput> getOutputs() {
      return outputs;
    }
  }

  /** An immutable result returned by a Classifier describing what was recognized. */
  public static class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  //------------------------- OLD CODE

//  List<Recognition> recognizeImage(final Bitmap bitmap);
//
//  List<Recognition> recognizeImage(final Bitmap bitmap, final float minimumConfidence);
//
//  List<Recognition> recognizeImage(final Bitmap bitmap, final int[][][] loadedIntValues);
//
//  void enableStatLogging(final boolean debug);
//
//  String getStatString();
//
//  void setNumThreads(int num_threads);
//
//  void setUseNNAPI(boolean isChecked);
}

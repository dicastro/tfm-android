/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.widget.Toast;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Device;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Model;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;

  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    tracker = new MultiBoxTracker(this);

    recreateClassifier(getModel(), getDevice(), getNumThreads(), getMinimumConfidence());

    if (detector == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
//    sensorOrientation = rotation - getScreenOrientation() - 90; // rotation is always 90, I do not understand why (this parameter is fixed and comes from CameraActivity class)
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(detector.getImageSizeX(), detector.getImageSizeY(), Config.ARGB_8888);

    frameToCropTransform = ImageUtils.getEnvelopeTransformationMatrix(previewWidth, previewHeight, detector.getImageSizeX(), detector.getImageSizeY());

//    frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight, detector.getImageSizeX(), detector.getImageSizeY(), sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = findViewById(R.id.tracking_overlay);

    trackingOverlay.addCallback(new DrawCallback() {
      @Override
      public void drawCallback(final Canvas canvas) {
        tracker.draw(canvas);
        if (isDebug()) {
          tracker.drawDebug(canvas);
        }
      }
    });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }

    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

//    try (InputStream stream = getAssets().open("g0010119.jpg")) {
//      rgbFrameBitmap = BitmapFactory.decodeStream(stream);
//    } catch (Exception ignored) {
//    }

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawColor(0xFF7F7F7F); // Gray
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

//    try (InputStream stream = getAssets().open("g0010119-preprocessed.jpg")) {
//      croppedBitmap = BitmapFactory.decodeStream(stream);
//    } catch (Exception ignored) {
//    }

    runInBackground(new Runnable() {
      @Override
      public void run() {
        LOGGER.i("Running detection on image " + currTimestamp);
        final long startTime = SystemClock.uptimeMillis();
        final List<Recognition> results = detector.recognizeImage(croppedBitmap);
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
        final Canvas canvas = new Canvas(cropCopyBitmap);

        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Style.STROKE);
        paint.setStrokeWidth(1.0f);

        final List<Recognition> mappedRecognitions = new LinkedList<>();

        for (final Recognition result : results) {
          final RectF location = result.getLocation();

          canvas.drawRect(result.getLocation(), paint);

          cropToFrameTransform.mapRect(location);

          result.setLocation(location);
          mappedRecognitions.add(result);
        }

        tracker.trackResults(mappedRecognitions, currTimestamp);
        trackingOverlay.postInvalidate();

        computingDetection = false;

        runOnUiThread(new Runnable() {
          @Override
          public void run() {
            showFrameInfo(previewWidth + "x" + previewHeight);
            showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
            showInference(lastProcessingTimeMs + "ms");
          }
        });
      }
    });
  }

  private void recreateClassifier(Model model, Device device, int numThreads, float minimumConfidence) {
    if (detector != null) {
      LOGGER.d("Closing classifier.");
      detector.close();
      detector = null;
    }

//    if (device == Device.GPU) {
//      LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
//      runOnUiThread(() -> Toast.makeText(this, "GPU does not fully supported.", Toast.LENGTH_LONG).show());
//      return;
//    }

    try {
      LOGGER.d("Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      detector = Classifier.create(this, model, device, numThreads, minimumConfidence);
    } catch (IOException e) {
      LOGGER.e(e, "Failed to create classifier.");
    }
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  protected void onInferenceConfigurationChanged() {
    if (croppedBitmap == null) {
      // Defer creation until we're getting camera frames.
      return;
    }

    final Device device = getDevice();
    final Model model = getModel();
    final int numThreads = getNumThreads();
    final float minimumConfidence = getMinimumConfidence();

    runInBackground(() -> recreateClassifier(model, device, numThreads, minimumConfidence));
  }
}

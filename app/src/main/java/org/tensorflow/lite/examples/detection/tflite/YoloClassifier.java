package org.tensorflow.lite.examples.detection.tflite;

import android.app.Activity;
import android.graphics.RectF;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public abstract class YoloClassifier extends Classifier {
    // Float model
    protected static final float IMAGE_MAX_VALUE = 255.0f;

    protected YoloClassifier(Activity activity, Device device, int numThreads, float minimumConfidence) throws IOException {
        super(activity, device, numThreads, minimumConfidence);
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

    protected List<Recognition> decodeNetout(final float[][][] netout, final int gridHeight, final int gridWidth, final int[] anchors, final float minimumConfidence) {
        int NUM_BOXES_PER_BLOCK = 3;
        int numClasses = getNumLabels();

        int netWidth = getImageSizeX();
        int netHeight = getImageSizeY();

        // Find the best detections.
        List<Recognition> detections = new ArrayList<>();

        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset = (numClasses + 5) * b;

                    final float confidence = expit(netout[y][x][offset + 4]);

                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[numClasses];
                    for (int c = 0; c < numClasses; ++c) {
                        classes[c] = netout[y][x][offset + 5 + c];
                    }
                    softmax(classes);

                    for (int c = 0; c < numClasses; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }

                    final float confidenceInClass = maxClass * confidence;
                    if (confidenceInClass >  minimumConfidence) {
                        final float xPos = (x + expit(netout[y][x][offset])) / gridWidth;
                        final float yPos = (y + expit(netout[y][x][offset + 1])) / gridHeight;

                        final float w = (float) (Math.exp(netout[y][x][offset + 2]) * anchors[2 * b]) / netWidth;
                        final float h = (float) (Math.exp(netout[y][x][offset + 3]) * anchors[2 * b + 1]) / netHeight;

                        final RectF rect = new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(netWidth - 1, xPos + w / 2),
                                Math.min(netHeight - 1, yPos + h / 2));

                        if (confidenceInClass > 0.01) {
                            detections.add(new Recognition("" + offset, getLabel(detectedClass), confidenceInClass, rect));
                        }
                    }
                }
            }
        }

        return detections;
    }

    protected void correctYoloBoxes(List<Recognition> recognitions) {
        int netWidth = getImageSizeX();
        int netHeight = getImageSizeY();
        int imageWidth = getImageSizeX();
        int imageHeight = getImageSizeY();

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

            RectF correctedLocation = new RectF(
                    (recognition.getLocation().left - x_offset) / x_scale * imageWidth,
                    (recognition.getLocation().top - y_offset) / y_scale * imageHeight,
                    (recognition.getLocation().right - x_offset) / x_scale * imageWidth,
                    (recognition.getLocation().bottom - y_offset) / y_scale * imageHeight);

            recognition.setLocation(correctedLocation);
        }
    }

    protected List<Recognition> doNms(List<Recognition> recognitions) {
        float NMS_THRESHOLD = 0.2f;

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

        Recognition currentPrediction;
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

                overlaps = overlaps || (intersectProportion > NMS_THRESHOLD);
            }

            if (!overlaps) {
                predictions.add(currentPrediction);
            }
        }

        return predictions;
    }
}

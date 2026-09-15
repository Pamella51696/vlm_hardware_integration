import org.opencv.core.Mat;

/**
 * Latest calibrated/stitched frame plus left/center/right ROIs.
 * Streaming and the VLM client both read this; the VLM is never fed every
 * camera frame — only the snapshot current when a query arrives.
 */
public final class FrameBuffer {

    public static final class Snapshot {
        public final long frameId;
        public final long timestampMs;
        public final byte[] panoramaJpeg;
        public final byte[] leftJpeg;
        public final byte[] centerJpeg;
        public final byte[] rightJpeg;
        public final int width;
        public final int height;

        Snapshot(long frameId, long timestampMs, byte[] panoramaJpeg,
                 byte[] leftJpeg, byte[] centerJpeg, byte[] rightJpeg,
                 int width, int height) {
            this.frameId = frameId;
            this.timestampMs = timestampMs;
            this.panoramaJpeg = panoramaJpeg;
            this.leftJpeg = leftJpeg;
            this.centerJpeg = centerJpeg;
            this.rightJpeg = rightJpeg;
            this.width = width;
            this.height = height;
        }

        public byte[] jpegFor(RoiSelector.View view) {
            switch (view) {
                case LEFT:
                    return leftJpeg;
                case CENTER:
                    return centerJpeg;
                case RIGHT:
                    return rightJpeg;
                case FULL:
                default:
                    return panoramaJpeg;
            }
        }
    }

    private final Object lock = new Object();
    private Snapshot latest;
    private long sequence;

    public void publish(Mat panorama) {
        if (panorama == null || panorama.empty()) {
            return;
        }
        Mat left = RoiSelector.crop(panorama, RoiSelector.View.LEFT);
        Mat center = RoiSelector.crop(panorama, RoiSelector.View.CENTER);
        Mat right = RoiSelector.crop(panorama, RoiSelector.View.RIGHT);
        try {
            byte[] fullJpeg = VideoStreamingServer.encodeJpeg(panorama);
            byte[] leftJpeg = VideoStreamingServer.encodeJpeg(left);
            byte[] centerJpeg = VideoStreamingServer.encodeJpeg(center);
            byte[] rightJpeg = VideoStreamingServer.encodeJpeg(right);
            synchronized (lock) {
                sequence++;
                latest = new Snapshot(sequence, System.currentTimeMillis(),
                        fullJpeg, leftJpeg, centerJpeg, rightJpeg,
                        panorama.cols(), panorama.rows());
                lock.notifyAll();
            }
        } finally {
            if (left != panorama) {
                left.release();
            }
            if (center != panorama) {
                center.release();
            }
            if (right != panorama) {
                right.release();
            }
        }
    }

    public Snapshot peek() {
        synchronized (lock) {
            return latest;
        }
    }

    public Snapshot waitForNext(long lastFrameId, long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        synchronized (lock) {
            while (latest == null || latest.frameId == lastFrameId) {
                long remaining = deadline - System.currentTimeMillis();
                if (remaining <= 0) {
                    return latest;
                }
                lock.wait(remaining);
            }
            return latest;
        }
    }
}

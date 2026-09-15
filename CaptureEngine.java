import java.nio.file.Path;

import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

/**
 * Single capture/stitch loop. Browser MJPEG and VLM queries share the same
 * undistorted panorama; the VLM is not invoked here.
 */
public final class CaptureEngine implements Runnable {

    private final Path[] videos;
    private final FrameBuffer buffer;
    private final boolean alignFrames;
    private final Thread thread;
    private volatile boolean running = true;

    public CaptureEngine(Path[] videos, FrameBuffer buffer) {
        this.videos = videos;
        this.buffer = buffer;
        this.alignFrames = !"0".equals(System.getenv().getOrDefault("VLM_ALIGN_FRAMES", "1"));
        this.thread = new Thread(this, "capture-engine");
        this.thread.setDaemon(true);
    }

    public void start() {
        thread.start();
    }

    public void stop() {
        running = false;
        thread.interrupt();
    }

    @Override
    public void run() {
        VideoCapture[] caps = new VideoCapture[videos.length];
        VideoStreamingServer.SphericalPanel[] panels =
                new VideoStreamingServer.SphericalPanel[videos.length];
        try {
            for (int i = 0; i < videos.length; i++) {
                caps[i] = VideoStreamingServer.openVideo(videos[i]);
                if (caps[i] == null || !caps[i].isOpened()) {
                    System.err.println("CaptureEngine: could not open " + videos[i]);
                    return;
                }
                panels[i] = new VideoStreamingServer.SphericalPanel(i);
            }

            Mat[] frames = new Mat[videos.length];
            Mat[] ready = new Mat[videos.length];
            for (int i = 0; i < videos.length; i++) {
                frames[i] = new Mat();
                ready[i] = new Mat();
            }
            OrbRansacAligner aligner = alignFrames ? new OrbRansacAligner() : null;
            int overlap = VideoStreamingServer.panelOverlapPx();

            while (running) {
                boolean allReady = true;
                for (int i = 0; i < caps.length; i++) {
                    if (!VideoStreamingServer.readOrLoop(caps, i, videos[i], frames[i])) {
                        allReady = false;
                        continue;
                    }
                    panels[i].project(frames[i], ready[i]);
                    if (ready[i].empty()
                            || ready[i].cols() != VideoStreamingServer.PANEL_WIDTH
                            || ready[i].rows() != VideoStreamingServer.PANEL_HEIGHT) {
                        allReady = false;
                    }
                }
                if (!allReady) {
                    continue;
                }
                Mat panorama = VideoStreamingServer.featherStitch(ready, overlap);
                Mat toPublish = panorama;
                if (aligner != null) {
                    Mat aligned = aligner.alignToPrevious(panorama);
                    if (aligned != panorama) {
                        panorama.release();
                        toPublish = aligned;
                    }
                }
                buffer.publish(toPublish);
                toPublish.release();
            }
        } catch (Exception e) {
            System.err.println("CaptureEngine stopped: " + e.getMessage());
            e.printStackTrace();
        } finally {
            for (VideoCapture c : caps) {
                if (c != null) {
                    c.release();
                }
            }
        }
    }
}

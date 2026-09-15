import org.opencv.core.Mat;
import org.opencv.core.Rect;

/**
 * Splits the undistorted/stitched panorama into LEFT / CENTER / RIGHT views
 * so a query such as "curb on the left" does not send the whole 360 canvas.
 */
public final class RoiSelector {

    public enum View {
        LEFT, CENTER, RIGHT, FULL
    }

    private RoiSelector() {}

    public static View fromQuery(String query) {
        if (query == null) {
            return View.FULL;
        }
        String q = query.toLowerCase();
        boolean left = q.contains("left") || q.contains("port");
        boolean right = q.contains("right") || q.contains("starboard");
        boolean ahead = q.contains("ahead") || q.contains("front")
                || q.contains("center") || q.contains("straight");
        if (left && !right) {
            return View.LEFT;
        }
        if (right && !left) {
            return View.RIGHT;
        }
        if (ahead && !left && !right) {
            return View.CENTER;
        }
        return View.FULL;
    }

    public static Mat crop(Mat panorama, View view) {
        if (panorama == null || panorama.empty() || view == View.FULL) {
            return panorama;
        }
        int w = panorama.cols();
        int h = panorama.rows();
        int third = Math.max(1, w / 3);
        int x0;
        int x1;
        switch (view) {
            case LEFT:
                x0 = 0;
                x1 = third;
                break;
            case CENTER:
                x0 = third;
                x1 = Math.min(w, 2 * third);
                break;
            case RIGHT:
                x0 = Math.min(w, 2 * third);
                x1 = w;
                break;
            default:
                return panorama;
        }
        if (x1 <= x0) {
            return panorama;
        }
        return new Mat(panorama, new Rect(x0, 0, x1 - x0, h));
    }
}

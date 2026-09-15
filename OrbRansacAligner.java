import java.util.ArrayList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

/**
 * Temporal alignment between consecutive undistorted frames.
 * This is NOT an object detector — it only estimates Frame N ↔ Frame N+1
 * motion so the buffer the VLM sees is more stable.
 */
public final class OrbRansacAligner {

    private static final int MAX_FEATURES = 900;
    private static final float RATIO = 0.75f;
    private static final int MIN_INLIERS = 18;
    private static final double RANSAC_REPROJ = 3.0;
    private static final double MAX_SHIFT = 48.0;

    private final ORB orb = ORB.create(MAX_FEATURES);
    private final DescriptorMatcher matcher =
            DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

    private Mat prevGray;
    private MatOfKeyPoint prevKp;
    private Mat prevDesc;

    public Mat alignToPrevious(Mat bgr) {
        if (bgr == null || bgr.empty()) {
            return bgr;
        }
        Mat gray = new Mat();
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        MatOfKeyPoint kp = new MatOfKeyPoint();
        Mat desc = new Mat();
        orb.detectAndCompute(gray, new Mat(), kp, desc);

        Mat aligned = bgr;
        if (prevGray != null && prevDesc != null && !prevDesc.empty()
                && desc != null && !desc.empty()) {
            Mat warped = tryWarp(bgr, gray, kp, desc);
            if (warped != null) {
                aligned = warped;
            }
        }

        if (prevGray != null) prevGray.release();
        if (prevKp != null) prevKp.release();
        if (prevDesc != null) prevDesc.release();
        prevGray = gray;
        prevKp = kp;
        prevDesc = desc;
        return aligned;
    }

    private Mat tryWarp(Mat bgr, Mat gray, MatOfKeyPoint kp, Mat desc) {
        List<MatOfDMatch> knn = new ArrayList<MatOfDMatch>();
        matcher.knnMatch(desc, prevDesc, knn, 2);
        List<Point> srcPts = new ArrayList<Point>();
        List<Point> dstPts = new ArrayList<Point>();
        KeyPoint[] now = kp.toArray();
        KeyPoint[] prev = prevKp.toArray();
        for (MatOfDMatch row : knn) {
            DMatch[] m = row.toArray();
            if (m.length < 2) {
                continue;
            }
            if (m[0].distance < RATIO * m[1].distance) {
                srcPts.add(now[m[0].queryIdx].pt);
                dstPts.add(prev[m[0].trainIdx].pt);
            }
            row.release();
        }
        if (srcPts.size() < MIN_INLIERS) {
            return null;
        }
        MatOfPoint2f src = new MatOfPoint2f();
        MatOfPoint2f dst = new MatOfPoint2f();
        src.fromList(srcPts);
        dst.fromList(dstPts);
        Mat mask = new Mat();
        Mat H = Calib3d.findHomography(src, dst, Calib3d.RANSAC, RANSAC_REPROJ, mask);
        src.release();
        dst.release();
        if (H == null || H.empty() || !plausible(H, mask)) {
            if (H != null) H.release();
            mask.release();
            return null;
        }
        Mat warped = new Mat();
        Imgproc.warpPerspective(bgr, warped, H, new Size(bgr.cols(), bgr.rows()),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
        H.release();
        mask.release();
        return warped;
    }

    private static boolean plausible(Mat H, Mat mask) {
        int inliers = Core.countNonZero(mask);
        if (inliers < MIN_INLIERS) {
            return false;
        }
        double[] h = new double[9];
        H.get(0, 0, h);
        double dx = Math.abs(h[2]);
        double dy = Math.abs(h[5]);
        if (dx > MAX_SHIFT || dy > MAX_SHIFT) {
            return false;
        }
        double det = h[0] * h[4] - h[1] * h[3];
        return det > 0.4 && det < 2.5;
    }
}

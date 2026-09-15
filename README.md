The Java stitcher still owns the cameras. A small Python process answers
scene questions on the *current undistorted frame only* — not every video frame.

    fisheye → calibrate/undistort → ORB/RANSAC align → frame buffer
         ├─ /stitch  (display, camera rate)
         └─ POST /ask → Python VLM (on demand) → JSON

Run the VLM service first:

    bash scripts/run_vlm.sh

Then compile and start the Java server the same way you already do, with every
`.java` file on the javac line, for example:

    javac -cp "$OPENCV_JAR" *.java
    java  -cp ".:$OPENCV_JAR" -Djava.library.path="$OPENCV_NATIVE" VideoStreamingServer

Open http://localhost:9090/play and use the query bar, or:

    curl -s -X POST http://localhost:9090/ask \
      -H 'Content-Type: application/json' \
      -d '{"query":"Is there a curb on the left side?"}'

Environment

    VLM_URL              Java → Python base URL (default http://127.0.0.1:8088)
    VLM_ALIGN_FRAMES     1 to enable ORB+RANSAC temporal align (default 1)
    VLM_BACKEND          hybrid (OpenCV priors) or transformers
    VLM_MODEL            Hugging Face id or local path for a 1B–3B-class VLM
    VLM_LOAD_4BIT        1 to request 4-bit weights on CUDA (Orin 8 GB)
    VLM_FORCE_TRANSFORMERS  1 to also load a VLM when backend=hybrid

Do not train a VLM first. Start with a pretrained quantized model that matches
your JetPack/CUDA/TensorRT stack, then fine-tune on *this* calibrated camera.

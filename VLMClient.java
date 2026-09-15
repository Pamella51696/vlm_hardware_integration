import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * Thin HTTP client from the Java video server to the Python VLM process.
 * The Java process never loads CUDA, tokenizers, or the vision encoder.
 */
public final class VLMClient {

    private final String askUrl;
    private final int connectTimeoutMs;
    private final int readTimeoutMs;

    public VLMClient() {
        this(System.getenv().getOrDefault("VLM_URL", "http://127.0.0.1:8088"),
                2000, 45_000);
    }

    public VLMClient(String baseUrl, int connectTimeoutMs, int readTimeoutMs) {
        String base = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.askUrl = base + "/ask";
        this.connectTimeoutMs = connectTimeoutMs;
        this.readTimeoutMs = readTimeoutMs;
    }

    public String sendQuery(String query, byte[] jpeg, long frameId, String roi,
                            String category, String hint) throws IOException {
        if (jpeg == null || jpeg.length == 0) {
            throw new IOException("No calibrated frame available for the VLM");
        }
        String boundary = "----VlmBoundary" + Long.toHexString(System.nanoTime());
        byte[] body = buildMultipart(boundary, query, jpeg, frameId, roi, category, hint);

        HttpURLConnection conn = (HttpURLConnection) new URL(askUrl).openConnection();
        conn.setDoOutput(true);
        conn.setRequestMethod("POST");
        conn.setConnectTimeout(connectTimeoutMs);
        conn.setReadTimeout(readTimeoutMs);
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        conn.setRequestProperty("Accept", "application/json");
        try (OutputStream out = conn.getOutputStream()) {
            out.write(body);
        }
        int status = conn.getResponseCode();
        InputStream in = status >= 400 ? conn.getErrorStream() : conn.getInputStream();
        String response = readAll(in);
        if (status >= 400) {
            throw new IOException("VLM service HTTP " + status + ": " + response);
        }
        return response;
    }

    public boolean healthy() {
        try {
            String base = askUrl.substring(0, askUrl.length() - "/ask".length());
            HttpURLConnection conn = (HttpURLConnection) new URL(base + "/health").openConnection();
            conn.setConnectTimeout(800);
            conn.setReadTimeout(800);
            conn.setRequestMethod("GET");
            return conn.getResponseCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    private static byte[] buildMultipart(String boundary, String query, byte[] jpeg,
                                         long frameId, String roi, String category,
                                         String hint) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        writeField(out, boundary, "query", query == null ? "" : query);
        writeField(out, boundary, "frame_id", Long.toString(frameId));
        writeField(out, boundary, "roi", roi == null ? "full" : roi);
        writeField(out, boundary, "category", category == null ? "other" : category);
        writeField(out, boundary, "hint", hint == null ? "" : hint);
        writeFile(out, boundary, "image", "frame.jpg", "image/jpeg", jpeg);
        out.write(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        return out.toByteArray();
    }

    private static void writeField(ByteArrayOutputStream out, String boundary,
                                   String name, String value) throws IOException {
        String part = "--" + boundary + "\r\n"
                + "Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n"
                + value + "\r\n";
        out.write(part.getBytes(StandardCharsets.UTF_8));
    }

    private static void writeFile(ByteArrayOutputStream out, String boundary,
                                  String name, String filename, String mime,
                                  byte[] data) throws IOException {
        String head = "--" + boundary + "\r\n"
                + "Content-Disposition: form-data; name=\"" + name
                + "\"; filename=\"" + filename + "\"\r\n"
                + "Content-Type: " + mime + "\r\n\r\n";
        out.write(head.getBytes(StandardCharsets.UTF_8));
        out.write(data);
        out.write("\r\n".getBytes(StandardCharsets.UTF_8));
    }

    private static String readAll(InputStream in) throws IOException {
        if (in == null) {
            return "";
        }
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        byte[] tmp = new byte[4096];
        int n;
        while ((n = in.read(tmp)) >= 0) {
            buf.write(tmp, 0, n);
        }
        return buf.toString("UTF-8");
    }
}

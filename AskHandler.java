import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

/**
 * POST /ask  { "query": "Is there a curb on the left side?" }
 * Uses the latest undistorted frame (and an optional ROI) — not the live 30 FPS stream.
 */
public final class AskHandler implements HttpHandler {

    private final FrameBuffer buffer;
    private final VLMClient client;

    public AskHandler(FrameBuffer buffer, VLMClient client) {
        this.buffer = buffer;
        this.client = client;
    }

    @Override
    public void handle(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1);
            return;
        }
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            send(ex, 405, jsonError("method not allowed"));
            return;
        }
        String body = readBody(ex);
        String query = extractJsonString(body, "query");
        if (query == null || query.trim().isEmpty()) {
            query = extractForm(body, "query");
        }
        if (query == null || query.trim().isEmpty()) {
            send(ex, 400, jsonError("missing query"));
            return;
        }
        query = query.trim();

        FrameBuffer.Snapshot snap = buffer.peek();
        if (snap == null || snap.panoramaJpeg == null) {
            send(ex, 503, jsonError("no calibrated frame yet; wait for capture"));
            return;
        }

        RoiSelector.View view = RoiSelector.fromQuery(query);
        String roiOverride = extractJsonString(body, "roi");
        if (roiOverride != null && !roiOverride.isEmpty()) {
            try {
                view = RoiSelector.View.valueOf(roiOverride.trim().toUpperCase());
            } catch (IllegalArgumentException ignored) {
            }
        }

        String category = QueryManager.categorize(query).name().toLowerCase();
        String hint = QueryManager.promptHint(query);
        try {
            String response = client.sendQuery(
                    query, snap.jpegFor(view), snap.frameId, view.name().toLowerCase(),
                    category, hint);
            send(ex, 200, response);
        } catch (IOException e) {
            send(ex, 502, jsonError("VLM service unavailable: " + e.getMessage()));
        }
    }

    static void addCors(HttpExchange ex) {
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        ex.getResponseHeaders().set("Access-Control-Allow-Methods", "POST, OPTIONS");
    }

    static String readBody(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody();
             ByteArrayOutputStream buf = new ByteArrayOutputStream()) {
            byte[] tmp = new byte[4096];
            int n;
            while ((n = in.read(tmp)) >= 0) {
                buf.write(tmp, 0, n);
            }
            return buf.toString("UTF-8");
        }
    }

    static void send(HttpExchange ex, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=UTF-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    static String jsonError(String message) {
        return "{\"error\":\"" + escape(message) + "\"}";
    }

    static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    /** Minimal extractor — avoids pulling in a JSON library. */
    static String extractJsonString(String body, String key) {
        if (body == null) {
            return null;
        }
        String needle = "\"" + key + "\"";
        int k = body.indexOf(needle);
        if (k < 0) {
            return null;
        }
        int colon = body.indexOf(':', k + needle.length());
        if (colon < 0) {
            return null;
        }
        int i = colon + 1;
        while (i < body.length() && Character.isWhitespace(body.charAt(i))) {
            i++;
        }
        if (i >= body.length() || body.charAt(i) != '"') {
            return null;
        }
        i++;
        StringBuilder sb = new StringBuilder();
        while (i < body.length()) {
            char c = body.charAt(i++);
            if (c == '\\' && i < body.length()) {
                sb.append(body.charAt(i++));
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    static String extractForm(String body, String key) {
        if (body == null) {
            return null;
        }
        for (String part : body.split("&")) {
            int eq = part.indexOf('=');
            if (eq > 0 && part.substring(0, eq).equals(key)) {
                return java.net.URLDecoder.decode(part.substring(eq + 1), "UTF-8");
            }
        }
        return null;
    }
}

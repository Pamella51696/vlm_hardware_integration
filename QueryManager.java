/**
 * Classifies user questions into the small vocabulary the VLM service handles.
 */
public final class QueryManager {

    public enum Category {
        ROAD, OBJECT, SPATIAL, SAFETY, OTHER
    }

    private QueryManager() {}

    public static Category categorize(String query) {
        if (query == null) {
            return Category.OTHER;
        }
        String q = query.toLowerCase();
        if (containsAny(q, "clear", "blocked", "curb", "sidewalk", "road", "lane")) {
            return Category.ROAD;
        }
        if (containsAny(q, "pedestrian", "person", "car", "vehicle", "bicycle",
                "motorcycle", "sign", "truck")) {
            return Category.OBJECT;
        }
        if (containsAny(q, "left", "right", "ahead", "where", "which side", "front")) {
            return Category.SPATIAL;
        }
        if (containsAny(q, "obstacle", "crossing", "approaching", "danger", "safe")) {
            return Category.SAFETY;
        }
        return Category.OTHER;
    }

    public static String promptHint(String query) {
        Category cat = categorize(query);
        switch (cat) {
            case ROAD:
                return "Focus on road surface, curbs, sidewalks, and whether the path is clear.";
            case OBJECT:
                return "Focus on counting and locating cars, pedestrians, bicycles, and signs.";
            case SPATIAL:
                return "Answer with left, right, center/ahead, or unknown.";
            case SAFETY:
                return "Prioritize obstacles, crossing people, and approaching vehicles.";
            default:
                return "Answer only from the image. Use the required JSON schema.";
        }
    }

    private static boolean containsAny(String q, String... keys) {
        for (String k : keys) {
            if (q.contains(k)) {
                return true;
            }
        }
        return false;
    }
}

import java.io.*;
import java.lang.reflect.Method;
import java.net.*;
import java.nio.file.*;
import java.util.Scanner;

public class Game {
    private static final String GSON_VERSION = "2.8.9";
    private static final String GSON_PATH = "gson.jar";
    private static final String GSON_URL = "https://repo1.maven.org/maven2/com/google/code/gson/gson/" + GSON_VERSION + "/gson-" + GSON_VERSION + ".jar";

    public static void main(String[] args) {
        try {
            checkAndDownloadGson();
            ClassLoader cl = loadGsonWithClassLoader();

            // Load story files and let the user choose
            File[] storyFiles = loadStoryFiles("story_");
            if (storyFiles == null || storyFiles.length == 0) {
                System.out.println("No story files found!");
                return;
            }

            File selectedFile = getUserSelectedFile(storyFiles);
            if (selectedFile == null) {
                System.out.println("Invalid selection made!");
                return;
            }

            System.out.println("Processing story from file: " + selectedFile.getName());
            processJsonStory(selectedFile, cl);  // Process the story using Gson from the dynamically loaded classloader
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("An error occurred during the game execution.");
        }
    }

    private static void checkAndDownloadGson() {
        Path gsonPath = Paths.get(GSON_PATH);
        if (Files.notExists(gsonPath)) {
            System.out.println("Downloading Gson...");
            try (InputStream in = new URL(GSON_URL).openStream()) {
                Files.copy(in, gsonPath, StandardCopyOption.REPLACE_EXISTING);
                System.out.println("Gson downloaded successfully.");
            } catch (IOException e) {
                e.printStackTrace();
                System.out.println("Failed to download Gson.");
            }
        }
    }

    private static ClassLoader loadGsonWithClassLoader() {
        try {
            URL gsonJarUrl = new File(GSON_PATH).toURI().toURL();
            return new URLClassLoader(new URL[]{gsonJarUrl});
        } catch (MalformedURLException e) {
            throw new RuntimeException("Error loading Gson JAR", e);
        }
    }

    private static File[] loadStoryFiles(String prefix) {
        File dir = new File(".");
        return dir.listFiles((dir1, name) -> name.startsWith(prefix) && name.endsWith(".json"));
    }

    private static File getUserSelectedFile(File[] files) {
        if (files.length == 1) return files[0];

        Scanner scanner = new Scanner(System.in);
        for (int i = 0; i < files.length; i++) {
            System.out.println((i + 1) + ": " + files[i].getName());
        }
        System.out.println("Select a story file by number:");

        int choice = scanner.nextInt() - 1;
        if (choice >= 0 && choice < files.length) {
            return files[choice];
        }
        System.out.println("Invalid selection. Please try again.");
        return null;
    }

    private static void processJsonStory(File storyFile, ClassLoader cl) {
        try {
            Class<?> gsonClass = cl.loadClass("com.google.gson.Gson");
            Object gson = gsonClass.getConstructor().newInstance();

            try (Reader reader = new FileReader(storyFile)) {
                Method fromJson = gsonClass.getMethod("fromJson", Reader.class, Class.forName("java.lang.Object"));
                Object storyObject = fromJson.invoke(gson, reader, Map.class);

                System.out.println("Story loaded: " + storyObject);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to process the story file.");
        }
    }
}

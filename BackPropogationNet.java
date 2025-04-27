
/**
 * Project 3: BackPropogation Neural Networks
 * Authors: Jonathan Rivera, Gianpaolo Tabora, Kian Drees, Precee-Noel Ginigeme
 * Class: COMP 380, Neural Networks
 * Prof: Eric Jiang
 * Due Date: 4/10/2025
 */
import java.util.Scanner;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BackPropogationNet {

    public static int rowDim;
    public static int colDim;



    public static void main(String[] args) {
        receiveInput();
    }

    public static void receiveInput() {
        Scanner scanner = new Scanner(System.in);

        int action = 0;
        do {
            if (action == 1) {
                trainingSpecs(scanner);
            } else if (action == 2) {
                testingSpecs(scanner);
            }
            System.out.println("Welcome to the LeafLens BackPropogation Neural Network.");
            System.out.println("1) Enter 1 to train the net on a data file");
            System.out.println("2) Enter 2 to test the net on a data file");
            System.out.println("3) Enter 3 to quit");
            action = scanner.nextInt();
        } while (action != 3);
        System.out.println("Thank you for using the Net. Come back soon!");
        scanner.close();
    }

    public static void trainingSpecs(Scanner scanner) {
        scanner.nextLine();
        System.out.println("Enter a file name to save the trained weight values:");
        String outputWeightFileName = scanner.nextLine();

        trainNetwork(outputWeightFileName);


    }

    public static void testingSpecs(Scanner scanner) {
        scanner.nextLine(); // get rid of newline read from previous nextInt()
        System.out.println("Enter the saved weights file name:");
        String savedWeightsFilename = scanner.nextLine();

        System.out.println("Enter the testing picture file name:");
        String testingDataFilename = scanner.nextLine();

        System.out.println("Enter the file name to save your results to:");
        String resultsFilename = scanner.nextLine();

        //testNetwork(savedWeightsFilename, testingDataFilename, resultsFilename);

    }

    public static ArrayList<Integer> readData(File file) {
        ArrayList<Integer> inputVector = new ArrayList<>();
        try {
            BufferedImage image = ImageIO.read(file);
            int width = image.getWidth();
            int height = image.getHeight();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixel = image.getRGB(x, y);

                    // Extract RGB components
                    int red = (pixel >> 16) & 0xff;
                    int green = (pixel >> 8) & 0xff;
                    int blue = pixel & 0xff;

                    // Simple brightness average
                    int brightness = (red + green + blue) / 3;

                    if (brightness < 128) { // Dark pixel = black
                        inputVector.add(1);
                    } else { // Light pixel = white
                        inputVector.add(-1);
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Failed to read image: " + file.getName());
        }
        return inputVector;
    }

    public static void gatherData(List<String> labels,List<ArrayList<Integer>> inputVectorsOfImages ) {
        File pestsFolder = new File("Pests"); // Assuming "Pests" folder is in project root

        if (!pestsFolder.exists() || !pestsFolder.isDirectory()) {
            System.err.println("Pests folder not found!");
            return;
        }

        for (File pestTypeFolder : pestsFolder.listFiles()) {
            if (pestTypeFolder.isDirectory()) {
                String pestName = pestTypeFolder.getName();

                for (File imageFile : pestTypeFolder.listFiles()) {
                    if (imageFile.isFile() && isImageFile(imageFile)) {
                        ArrayList<Integer> inputVector = readData(imageFile);
                        labels.add(pestName);
                        inputVectorsOfImages.add(inputVector);
                    }
                }
            }
        }
    }

    // Helper method to check if it's a PNG image
    private static boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg");
    }



    public static void trainNetwork(String fileToWrite) {
        List<String> labels = new ArrayList<>();
         List<ArrayList<Integer>> globalData = new ArrayList<>();
         gatherData(labels,globalData);
    }

    public static void writeWeightMatrixToFile(String fileToWrite, int[][] weightMatrix) {
        /**
         * Writes the given weight matrix to a text file in a space-separated format,
         * where each row of the matrix is written as a separate line in the file.
         *
         * @param fileToWrite  the path of the file to write the weight matrix to
         * @param weightMatrix a 2D integer array representing the weight matrix
         */
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileToWrite))) {
            int size = weightMatrix.length;

            writer.write(rowDim + " //row length");
            writer.newLine();
            writer.write(colDim + " //col length");
            writer.newLine();
            writer.newLine();

            for (int i = 0; i < size; i++) {
                StringBuilder row = new StringBuilder();
                for (int j = 0; j < size; j++) {
                    row.append(weightMatrix[i][j]);
                    if (j < size - 1) {
                        row.append(" ");
                    }
                }
                writer.write(row.toString());
                writer.newLine();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static int[][] readWeightMatrixFromFile(String filePath) {
        /**
         * Reads a weight matrix from a text file where each line represents a row of
         * space-separated integers. This method reconstructs and returns the matrix
         * as a 2D integer array.
         *
         * @param filePath the path to the file containing the saved weight matrix
         * @return a 2D integer array representing the weight matrix read from the file
         */
        List<int[]> rows = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            rowDim = Integer.parseInt(reader.readLine().trim().split("\\s+")[0]);
            colDim = Integer.parseInt(reader.readLine().trim().split("\\s+")[0]);
            reader.readLine();

            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");
                int[] row = new int[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    row[i] = Integer.parseInt(tokens[i]);
                }
                rows.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Convert list to 2D array
        return rows.toArray(new int[rows.size()][]);
    }

    public static void testNetwork(String savedWeightsFilename, String testingDataFilename, String resultsFilename) {


    }

    /**
     * Helper function to apply the activation function
     *
     * @param y_in
     * @param previous_y_i
     * @return 1, -1, or previous_y_i
     */
    public static int activationFunction(int y_in, int previous_y_i) {
        return 0;
    }
}


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
        scanner.nextLine(); // get rid of newline read from previous nextInt()
        System.out.println("Enter the training data folder name:");
        String trainingDataFileName = scanner.nextLine();

        System.out.println("Enter a file name to save the trained weight values:");
        String outputWeightFileName = scanner.nextLine();

        //trainNetwork(outputWeightFileName, trainingDataFileName);

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

    public static ArrayList<int[]> readData(String filename) {

        return null;

    }

    public static void trainNetwork(String fileToWrite, String trainingData) {

        
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

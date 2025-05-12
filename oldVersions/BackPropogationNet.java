package oldVersions;

/**
 * Project 3: BackPropogation Neural Networks
 * Authors: Jonathan Rivera, Gianpaolo Tabora, Kian Drees, Precee-Noel Ginigeme
 * Class: COMP 380, Neural Networks
 * Prof: Eric Jiang
 * Due Date: 4/10/2025
 */
import java.util.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BackPropogationNet {

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

        System.out.println("Enter the testing picture folder name:");
        String testingDataFilename = scanner.nextLine();

        System.out.println("Enter the file name to save your results to:");
        String resultsFilename = scanner.nextLine();

        testNetwork(savedWeightsFilename, testingDataFilename, resultsFilename);

    }

    public static ArrayList<Integer> readData(File file) {
        ArrayList<Integer> inputVector = new ArrayList<>();
        try {
            BufferedImage originalImage = ImageIO.read(file);

            // Scale the image to 500x500
            BufferedImage scaledImage = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB);
            scaledImage.getGraphics().drawImage(originalImage, 0, 0, 500, 500, null);

            int width = scaledImage.getWidth();
            int height = scaledImage.getHeight();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixel = scaledImage.getRGB(x, y);

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

    public static void gatherData(String foldername, List<String> labels,
            List<ArrayList<Integer>> inputVectorsOfImages) {
        File trainingDataFolder = new File(foldername); // Assuming "Training_data" folder is in project root

        if (!trainingDataFolder.exists() || !trainingDataFolder.isDirectory()) {
            System.err.println("Pests folder not found!");
            return;
        }

        for (File pestTypeFolder : trainingDataFolder.listFiles()) {
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
        List<ArrayList<Integer>> inputVectorsOfImages = new ArrayList<>();
        gatherData("Training_data", labels, inputVectorsOfImages);

        // alpha
        double alpha = 0.5;

        // map rabbit will 1000 grasshopper 0100 rat 0010 snail 0001
        Map<String, Integer[]> labelMap = new HashMap<>();

        labelMap.put("Grasshopper", new Integer[] { 1, 0, 0, 0 });
        labelMap.put("Rabbit", new Integer[] { 0, 1, 0, 0 });
        labelMap.put("Rat", new Integer[] { 0, 0, 1, 0 });
        labelMap.put("Snail", new Integer[] { 0, 0, 0, 1 });

        double[][] weightMatrixV = new double[250000][2000]; // i by j
        double[][] weightMatrixW = new double[2000][4]; // j by k
        // initialize weights to random values between -0.5 and 0.5 later

        // error
        double sumOfErrors;

        boolean converged = false;
        // initialize several arrays for performance
        double[] inputLayer = new double[250000];
        double[] hiddenLayerIn = new double[2000];
        double[] hiddenLayer = new double[2000];
        double[] outputLayerIn = new double[4];
        double[] outputLayer = new double[4];
        double[] deltak = new double[4];
        double[] deltainj = new double[2000];
        double[] deltaj = new double[2000];

        while (!converged) {
            sumOfErrors = 0;
            // step 2 for each training pair, do 3-8
            for (int p = 0; p < inputVectorsOfImages.size(); p++) {
                // feedforward
                // step 3 each input receives input signal and broadcasts signal to all units in
                // hidden layer
                for (int i = 0; i < 250000; i++) {
                    inputLayer[i] = inputVectorsOfImages.get(p).get(i);
                }

                // step 4 each hidden unit sums its weighted input signals, applies its
                // activation function
                // to compute its output signal, and sends this signal to all units in the layer
                // above
                for (int j = 0; j < 2000; j++) {
                    double sum = 0;
                    for (int i = 0; i < 250000; i++) {
                        sum += inputLayer[i] * weightMatrixV[i][j];
                    }
                    hiddenLayerIn[j] = sum;
                    hiddenLayer[j] = activationFunction(sum); // Assuming activationFunction(y_in, previous_y_i)
                }

                // step 5 each output unit sums weighted input signals and applies activation
                // function to compute output signal

                for (int k = 0; k < 4; k++) {
                    double sum = 0;
                    for (int j = 0; j < 2000; j++) {
                        sum += hiddenLayer[j] * weightMatrixW[j][k];
                    }
                    outputLayerIn[k] = sum;
                    outputLayer[k] = activationFunction(sum);
                }

                // backpropogation of error
                // step 6 each output unit receives a target pattern corresponding to input
                // training pattern, computes its
                // error information term, then calculates its weight correction term and bias
                // correction term,and sends error term to units in layer below

                for (int k = 0; k < 4; k++) {
                    deltak[k] = (labelMap.get(labels.get(p))[k] - outputLayer[k])
                            * activationFunctionDeriv(outputLayerIn[k]);
                    // calculate sum of error (square?) for stopping condition
                    sumOfErrors += deltak[k];
                }
                // weight correction term below

                // step 7 each hidden unit sums its delta inputs, multiplies by derivative of
                // its activation function to calculate its error information term,
                // and calculates its weights correction term and bias correction term

                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 2000; j++) {
                        deltainj[j] = deltak[k] * weightMatrixW[j][k];
                    }
                }

                for (int j = 0; j < 2000; j++) {
                    deltaj[j] = deltainj[j] * activationFunctionDeriv(hiddenLayerIn[j]);
                    // calculate sum of error (square?) for stopping condition
                    sumOfErrors += deltaj[j];

                }
                // step 8 each output unit updates its bias and weights , and each hidden unit
                // updates its bias and weights (distributed)

                // weight correction term and updates weights
                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 2000; j++) {
                        // new = old + err correction term
                        weightMatrixW[j][k] = weightMatrixW[j][k] + alpha * deltak[k] * hiddenLayer[j];
                    }
                }
                // weight correction term and updates weights
                for (int j = 0; j < 2000; j++) {
                    for (int i = 0; i < 250000; i++) {
                        // new = old + err correction term
                        weightMatrixV[i][j] = weightMatrixV[i][j] + alpha * deltaj[j] * inputLayer[i];
                    }
                }
            }
            // step 9, test stopping condition (stop if error goes below threshold)
            System.out.println(sumOfErrors);
            if (sumOfErrors < 10) {
                converged = true;
            }

        }

        // save both weight matrices along with biases

        writeWeightMatrixToFile(fileToWrite, weightMatrixV, weightMatrixW);
    }

    public static void writeWeightMatrixToFile(String fileToWrite, double[][] weightMatrixV, double[][] weightMatrixW) {
        /**
         * Writes the given weight matrix to a text file in a space-separated format,
         * where each row of the matrix is written as a separate line in the file.
         *
         * @param fileToWrite  the path of the file to write the weight matrix to
         * @param weightMatrix a 2D integer array representing the weight matrix
         */
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileToWrite))) {
            int sizeI = weightMatrixV.length;// 250000
            int sizeJ = weightMatrixW.length;// 2000
            int sizeK = weightMatrixW[0].length;// 4

            writer.write(sizeI + " //i length");
            writer.newLine();
            writer.write(sizeJ + " //j length");
            writer.newLine();
            writer.write(sizeK + " //k length");
            writer.newLine();
            writer.write("Weight Matrix V\n");

            // write biases

            for (int i = 0; i < sizeI; i++) {
                StringBuilder row = new StringBuilder();
                for (int j = 0; j < sizeJ; j++) {
                    row.append(weightMatrixV[i][j]);
                    if (j < sizeJ - 1) {
                        row.append(" ");
                    }
                }
                writer.write(row.toString());
                writer.newLine();
            }
            writer.newLine();
            writer.write("Weight Matrix W");

            for (int j = 0; j < sizeJ; j++) {
                writer.newLine();
                StringBuilder row = new StringBuilder();
                for (int k = 0; k < sizeK; k++) {
                    row.append(weightMatrixV[j][k]);
                    if (k < sizeK - 1) {
                        row.append(" ");
                    }
                }
                writer.write(row.toString());

            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static WeightContainer readWeightMatrixFromFile(String filePath) {
        /**
         * Reads a weight matrix from a text file where each line represents a row of
         * space-separated integers. This method reconstructs and returns the matrix
         * as a 2D integer array.
         *
         * @param filePath the path to the file containing the saved weight matrix
         * @return a 2D integer array representing the weight matrix read from the file
         */
        List<double[]> rows = new ArrayList<>();
        WeightContainer wc = new WeightContainer();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            int sizeI = Integer.parseInt(reader.readLine().trim().split("\\s+")[0]);
            int sizeJ = Integer.parseInt(reader.readLine().trim().split("\\s+")[0]);
            int sizeK = Integer.parseInt(reader.readLine().trim().split("\\s+")[0]);
            reader.readLine();

            String line;
            while (true) {
                line = reader.readLine();
                if (line.isEmpty()) {
                    break;
                }
                String[] tokens = line.trim().split("\\s+");
                double[] row = new double[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    row[i] = Double.parseDouble(tokens[i]);
                }
                rows.add(row);
            }
            double[][] weightMatrixV = rows.toArray(new double[rows.size()][]);
            rows.clear();

            reader.readLine();

            while ((line = reader.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");
                double[] row = new double[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    row[i] = Double.parseDouble(tokens[i]);
                }
                rows.add(row);
            }
            double[][] weightMatrixW = rows.toArray(new double[rows.size()][]);

            wc.setWeights(weightMatrixV, weightMatrixW);

        } catch (IOException e) {
            e.printStackTrace();
        }

        // Convert list to 2D array
        return wc;
    }

    public static void testNetwork(String savedWeightsFilename, String testingDataFilename, String resultsFilename) {
        // get weights from file
        WeightContainer wc = readWeightMatrixFromFile(savedWeightsFilename);

        double[][] weightMatrixV = wc.getWeightMatrixV();
        double[][] weightMatrixW = wc.getWeightMatrixW();

        List<String> labels = new ArrayList<>();
        List<ArrayList<Integer>> inputVectorsOfImages = new ArrayList<>();
        gatherData(testingDataFilename, labels, inputVectorsOfImages);

        // map rabbit will 1000 grasshopper 0100 rat 0010 snail 0001
        Map<Integer[], String> labelMap = new HashMap<>();

        labelMap.put(new Integer[] { 1, 0, 0, 0 }, "Grasshopper");
        labelMap.put(new Integer[] { 0, 1, 0, 0 }, "Rabbit");
        labelMap.put(new Integer[] { 0, 0, 1, 0 }, "Rat");
        labelMap.put(new Integer[] { 0, 0, 0, 1 }, "Snail");

        double[] inputLayer = new double[250000];
        double[] hiddenLayer = new double[2000];
        double[] outputLayer = new double[4];
        Integer[] outputLayerInts = new Integer[4];

        StringBuilder printString = new StringBuilder();

        for (int p = 0; p < inputVectorsOfImages.size(); p++) {
            // feedforward
            // step 3 each input receives input signal and broadcasts signal to all units in
            // hidden layer
            for (int i = 0; i < 250000; i++) {
                inputLayer[i] = inputVectorsOfImages.get(p).get(i);
            }

            // step 4 each hidden unit sums its weighted input signals, applies its
            // activation function
            // to compute its output signal, and sends this signal to all units in the layer
            // above
            for (int j = 0; j < 2000; j++) {
                double sum = 0;
                for (int i = 0; i < 250000; i++) {
                    sum += inputLayer[i] * weightMatrixV[i][j];
                }
                hiddenLayer[j] = activationFunction(sum); // Assuming activationFunction(y_in, previous_y_i)
            }

            // step 5 each output unit sums weighted input signals and applies activation
            // function to compute output signal

            for (int k = 0; k < 4; k++) {
                double sum = 0;
                for (int j = 0; j < 2000; j++) {
                    sum += hiddenLayer[j] * weightMatrixW[j][k];
                }
                outputLayer[k] = activationFunction(sum);

                // convert almost 1 to 1 and else to 0
                if (outputLayer[k] > .8) { // change this number to change level of confidence to spray at
                    outputLayerInts[k] = 1;
                } else {
                    outputLayerInts[k] = 0;
                }

            }

            if (labelMap.containsKey(outputLayerInts)) {
                printString.append(labels.get(p)).append(" identified as ").append(labelMap.get(outputLayerInts))
                        .append("\n");
            } else {
                printString.append(labels.get(p)).append(" was not recognized as a pest").append("\n");
            }
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFilename))) {
            writer.write(printString.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * Helper function to apply the activation function
     *
     * @param y_in
     * @return 1, -1, or previous_y_i
     */
    public static double activationFunction(double y_in) {
        // bipolar sigmoid
        return (2.0 / (1.0 + Math.exp(-y_in))) - 1.0;
    }

    public static double activationFunctionDeriv(double y_in) {
        return 0.5 * (1.0 + activationFunction(y_in)) * (1.0 - activationFunction(y_in));
    }
}
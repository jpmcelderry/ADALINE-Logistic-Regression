import java.util.*;
import java.io.*;

public class ML4 {
	public static void main(String[] args) throws Exception {
		ArrayList<ArrayList<classificationSample>> data = new ArrayList<ArrayList<classificationSample>>();
		for(String file:args) {
			switch(file) {
			case "breast-cancer-wisconsin.data":
				data = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(data, new int[][]{{}},.005,false,file);
				break;
			case "glass.data":
				data = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(data, new int[][]{{}},.01,false,file);
				break;
			case "house-votes-84.data":
				data = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(data, new int[][]{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}},.01,false,file);
				break;
			case "iris.data":
				data = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(data, new int[][]{{}},.01,false,file);
				break;
			case "soybean-small.data":
				data = partitionData.partitionClassificationData(readData.readClassificationData(file));
				fiveFoldClassify(data, new int[][]{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34},
					{7,2,3,3,2,4,4,3,3,3,2,2,3,3,3,2,2,3,2,2,4,4,2,3,2,3,2,4,5,2,2,2,2,2,3}},.01,true,file);
				break;
			default:
				throw new IllegalArgumentException("Unrecognized classification file: " + file);
			}
		}
	}
	
	/*
	 * Driver method to classify an input dataset with five-fold classification
	 */
	public static void fiveFoldClassify(ArrayList<ArrayList<classificationSample>> samples, int[][] categoricalFeatures, double n, boolean printOutput, String fileName) throws IOException {
		BufferedWriter buffWriter = null;
		try{
			buffWriter = new BufferedWriter(new FileWriter(fileName + ".out.txt"));
			double[] logRegression = new double[5];
			double[] adaline = new double[5];
			
			//z-score transform the data
			samples = helpers.zScoreTransform(samples,helpers.booleanizeCategories(samples.get(0).get(0).features.length,categoricalFeatures[0]));

			//iterate for all 5 folds
			for (int holdOut=0;holdOut<5;holdOut++) {
				ArrayList<classificationSample> trainingData = new ArrayList<classificationSample>();
				//consolidate training folds into one dataset
				for(int trainingFold=0;trainingFold<5;trainingFold++) {
					if(trainingFold != holdOut) {
						for(classificationSample sample: samples.get(trainingFold)) {
							trainingData.add(sample);
						}
					}
				}
				boolean print=false;
				if(printOutput && holdOut==0) {
					print=true;
				}
				//train and test adaline model
				adaline adalineModel = new adaline();
				adalineModel.train(trainingData,categoricalFeatures,n,print);
				String[] adalinePredictions = adalineModel.test(samples.get(holdOut),print);
				adaline[holdOut]=calc01Loss(adalinePredictions,samples.get(holdOut));
				if(print) {
					for(int index=0;index<adalinePredictions.length;index++) {
						System.out.print("PREDICTED: " + adalinePredictions[index] + ", ACTUAL: " + samples.get(holdOut).get(index).classification + "\n");
					}
				}
				
				//train and test logistic regression model
				logisticRegression logRegress = new logisticRegression();
				logRegress.train(trainingData,categoricalFeatures,n,print);
				String[] logPredictions = logRegress.test(samples.get(holdOut),print);
				logRegression[holdOut]=calc01Loss(logPredictions,samples.get(holdOut));
				if(print) {
					for(int index=0;index<logPredictions.length;index++) {
						System.out.print("PREDICTED: " + logPredictions[index] + ", ACTUAL: " + samples.get(holdOut).get(index).classification + "\n");
					}
				}
				
				buffWriter.write("--------------HOLD OUT SET: " + (holdOut+1) + "---------------\n");
				buffWriter.write("Adaline Error %: " + (adaline[holdOut]*100) + "\n");
				buffWriter.write("Logistic Regression Error %: " + (logRegression[holdOut]*100) + "\n\n");

			}
			double adalineAvg = 0;
			double logRegressAvg = 0;
			for(int index=0;index<5;index++) {
				adalineAvg += adaline[index]/5.0;
				logRegressAvg += logRegression[index]/5.0;
			}
			buffWriter.write("--------------AVERAGE PERFORMANCE---------------\n");
			buffWriter.write("Adaline Error %: " + (adalineAvg*100) + "\nLogistic Regression Error %: " + (logRegressAvg*100) + "\n\n");
		}
		finally {
			if(buffWriter!=null) {buffWriter.close();}
		}
	}
	
	/*
	 * Method to calculate 0-1 loss between predictions and real data
	 */
	public static double calc01Loss(String[] predictions, ArrayList<classificationSample> actual) {
		int errors = 0;
		for(int index=0;index<predictions.length;index++) {
			if(!predictions[index].equals(actual.get(index).classification)) {
				errors++;
			}
		}
		return (double) errors/predictions.length;
	}
}

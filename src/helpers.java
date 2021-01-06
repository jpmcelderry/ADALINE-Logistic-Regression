import java.util.*;

public class helpers {
	/*
	 * Method to check whether an array contains a given integer, returns the index where that item is found
	 */
	public static int arrayContains(int[] array, int query) {
		for (int index=0;index<array.length;index++) {
			if (array[index]==query) {
				return index;
			}
		}
		return -1;
	}
	
	public static String[] randomImpute(ArrayList<String[]> data, String[] toImpute, int classIndex, int attributeIndex) {
		List<String> list = new ArrayList<>();
		//iterate over the whole array to find samples with matching class as the data point to impute
		for (String[] sample: data) {
			//If the samples match and the other sample has a value, add it to the list
			if(sample[classIndex].equals(toImpute[classIndex]) && !sample[attributeIndex].equals("?")) {
				list.add(sample[attributeIndex]);
			}
		}
		//Only randomly select a value if the list isn't empty
		if(list.size()>0) {
			Random rand = new Random();
			String randomVal = list.get(rand.nextInt(list.size()));
			toImpute[attributeIndex] = randomVal;
		}
		//If no values were extracted, return the item as is
		else {
			return toImpute;
		}
		return toImpute;
	}
	
	public static boolean[] booleanizeCategories(int numFeatures, int[] categoricalFeatures) {
		boolean[] returnArray = new boolean[numFeatures];
		for(int index=0;index<numFeatures;index++) {
			if(arrayContains(categoricalFeatures, index)!=-1) {
				returnArray[index]=true;
			}
			else {
				returnArray[index]=false;
			}
		}
		return returnArray;
	}
	
	public static ArrayList<ArrayList<classificationSample>> zScoreTransform(ArrayList<ArrayList<classificationSample>> samples, boolean[] isCategorical) {
		double[] means = new double[samples.get(0).get(0).features.length];
		double[] standardDev = new double[samples.get(0).get(0).features.length];
		double sampleCount = 0;
		for(ArrayList<classificationSample> sampleSet: samples) {
			sampleCount += sampleSet.size();
		}
		//Calculate the mean of each feature
		for(ArrayList<classificationSample> set: samples) {
			for(classificationSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(isCategorical[feature]==false) {
						means[feature] += sample.features[feature]/sampleCount;
					}
				}
			}
		}
		//Finds variance from sum of squared difference
		for(ArrayList<classificationSample> set: samples) {
			for(classificationSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(isCategorical[feature]==false) {
						double squareDif = (sample.features[feature]-means[feature])*(sample.features[feature]-means[feature]);
						standardDev[feature] += squareDif/sampleCount;
					}
				}
			}
		}
		//Find standard deviation from variance
		for(int feature=0;feature<means.length;feature++) {
			if(isCategorical[feature]==false) {
				standardDev[feature] = Math.sqrt(standardDev[feature]);
			}
		}
		//Z-score normalization
		for(ArrayList<classificationSample> set: samples) {
			for(classificationSample sample: set) {
				for(int feature=0;feature<sample.features.length;feature++) {
					if(isCategorical[feature]==false) {
						sample.features[feature] = (double) (sample.features[feature]-means[feature])/standardDev[feature];
					}
				}
			}
		}
		
		return samples;
	}
	
	/*
	 * Method to copy a 2D array
	 */
	public static double[][] clone(double[][] input){
		double[][] clone = new double[input.length][];
		for(int index1=0;index1<input.length;index1++) {
			clone[index1]=new double[input[index1].length];
			for(int index2=0;index2<input[index1].length;index2++) {
				clone[index1][index2] = input[index1][index2];
			}
		}
		return clone;
	}
	
	/*
	 * Method to print the weight vectors
	 */
	public static void printWeights(ArrayList<weightVector> weights) {
		for (int classIndex=0;classIndex<weights.size();classIndex++) {
			weightVector vector = weights.get(classIndex);
			System.out.println("---------WEIGHT VECTOR, CLASS " + weights.get(classIndex).classID + "----------");
			for(int feature=0;feature<vector.weights.length;feature++) {
				System.out.print("Feature " + feature + ": ");
				for(int value=0;value<vector.weights[feature].length;value++) {
					System.out.print(value + ": " + vector.weights[feature][value] + "\t");
				}
				System.out.println();
			}
		}
	}
}

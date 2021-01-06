import java.util.*;

public class logisticRegression {

	public ArrayList<weightVector> weightVectors;
	
	logisticRegression(){
		this.weightVectors = new ArrayList<weightVector>();
	}
	
	/*
	 * Training method via gradient descent
	 */
	public void train(ArrayList<classificationSample> trainingData, int[][] categoricalFeatures, double inputN, boolean printOutput) {
		
		//create an empty template that works for the dataset
		//if a feature is not categorical, then just create one weight
		//if a feature is categorical, create an individual weight for each category
		double[][] template = new double[trainingData.get(0).features.length+1][];
		for(int feature=0;feature<template.length;feature++) {
			int categoricalCheck = helpers.arrayContains(categoricalFeatures[0],feature);
			if (categoricalCheck!=-1) {
				template[feature] = new double[categoricalFeatures[1][categoricalCheck]];
			}
			else {
				template[feature] = new double[1];
			}
		}
		
		//Create a weight vector for each observed class
		HashMap<String,Integer> classesSeen = new HashMap<String,Integer>();
		for(classificationSample sample: trainingData) {
			if(classesSeen.get(sample.classification)==null) {
				weightVectors.add(new weightVector(sample.classification, helpers.clone(template)));
				classesSeen.put(sample.classification,1);
			}
		}
		
		//Instantiate random weights
		for(weightVector vector: weightVectors) {
			for (int feature=0;feature<vector.weights.length;feature++) {
				for(int value=0;value<vector.weights[feature].length;value++) {
					vector.weights[feature][value] = (Math.random() * .02)-.01;
				}
			}
		}
		
		boolean[] categoricalGuide = helpers.booleanizeCategories(trainingData.get(0).features.length,categoricalFeatures[0]);
		
		//instantiate variables for the while loop
		int prevMisclassifications = Integer.MAX_VALUE;
		int misclassifications=0;
		ArrayList<double[][]> weightUpdates = new ArrayList<double[][]>();
		for(int counter=0;counter<weightVectors.size();counter++) {
			weightUpdates.add(helpers.clone(template));
		}
		
		int iterations=0;
		//train until misclassifications don't decrease from one run to the next
		while(iterations<500) {
			//Reset the weight updates to 0
			for(double[][] updateVector:weightUpdates) {
				for(int feature=0;feature<updateVector.length;feature++) {
					for(int value=0;value<updateVector[feature].length;value++) {
						updateVector[feature][value] = 0;
					}
				}
			}
			misclassifications=0;
			
			//main loop
			for(classificationSample sample: trainingData) {
				double[] classProbs = new double[weightVectors.size()];
				for(int classIndex=0;classIndex<weightVectors.size();classIndex++) {
					weightVector vector = weightVectors.get(classIndex);
					//calculate the regression
					double output = 0;
					for(int feature=0;feature<sample.features.length;feature++) {
						if(categoricalGuide[feature]==true) {
							output += vector.weights[feature][(int)sample.features[feature]];
						}
						else {
							output += sample.features[feature]*vector.weights[feature][0];
						}
					}
					//add bias and exponentiate
					output += vector.weights[vector.weights.length-1][0];
					classProbs[classIndex]=Math.exp(output);
				}
				double denominator = 0;
				for(double output: classProbs) {
					denominator += output;
				}
				for(int index=0;index<classProbs.length;index++) {
					classProbs[index] = classProbs[index]/denominator;
				}
				//update the update values
				for(int classIndex=0;classIndex<weightUpdates.size();classIndex++) {
					for(int feature=0;feature<sample.features.length;feature++) {
						boolean print = false;
						if(printOutput && feature==0) {
							print = true;
						}
						if(categoricalGuide[feature]==true) {
							weightUpdates.get(classIndex)[feature][(int)sample.features[feature]] += calcWeightUpdate(sample.classification,weightVectors.get(classIndex).classID,classProbs[classIndex],1,print);
						}
						else {
							weightUpdates.get(classIndex)[feature][0] += calcWeightUpdate(sample.classification,weightVectors.get(classIndex).classID,classProbs[classIndex],sample.features[feature],print);
						}
					}
					//update bias
					weightUpdates.get(classIndex)[weightUpdates.get(classIndex).length-1][0] += calcWeightUpdate(sample.classification,weightVectors.get(classIndex).classID,classProbs[classIndex],1,false);
				}
				//make a prediction
				int prediction = 0;
				for(int index=1;index<classProbs.length;index++) {
					if(classProbs[index]>classProbs[prediction]) {
						prediction=index;
					}
				}
				if(!weightVectors.get(prediction).classID.equals(sample.classification)) {
					misclassifications++;
				}
			}
			if(misclassifications > prevMisclassifications) {
				break;
			}
			double n = inputN*((double) misclassifications/trainingData.size());
			//Apply the weight updates
			for(int classIndex=0;classIndex<weightUpdates.size();classIndex++) {
				if(printOutput) {
					System.out.println("---------UPDATE FOR CLASS " + weightVectors.get(classIndex).classID + "----------");
				}
				for(int feature=0;feature<weightUpdates.get(classIndex).length;feature++) {
					if(printOutput) {
						System.out.print("Feature " + feature + ": ");
					}
					for(int value=0;value<weightUpdates.get(classIndex)[feature].length;value++) {
						if(printOutput) {
							System.out.print(value + ": " + weightUpdates.get(classIndex)[feature][value] + "\t");
						}
						weightVectors.get(classIndex).weights[feature][value] += n*weightUpdates.get(classIndex)[feature][value];
					}
					if(printOutput) {
						System.out.println();
					}
				}
			}
			if(printOutput) {
				System.out.println("Misclassified " + misclassifications + " samples.");
			}
			prevMisclassifications=misclassifications;
			iterations++;
		}
		if(printOutput) {
			helpers.printWeights(weightVectors);
		}
	}
	
	/*
	 * Testing method, determines class by the highest posterior probability
	 */
	public String[] test(ArrayList<classificationSample> samples, boolean printOutput) {
		String[] predictions = new String[samples.size()];
		
		for(int sampleIndex=0;sampleIndex<samples.size();sampleIndex++) {
			classificationSample sample = samples.get(sampleIndex);
			double[] classProbs = new double[weightVectors.size()];
			for(int classIndex=0;classIndex<weightVectors.size();classIndex++) {
				weightVector vector = weightVectors.get(classIndex);
				//calculate the regression
				double output = 0;
				for(int feature=0;feature<sample.features.length;feature++) {
					if(vector.weights[feature].length>1) {
						output += vector.weights[feature][(int)sample.features[feature]];
					}
					else {
						output += sample.features[feature]*vector.weights[feature][0];
					}
				}
				//add bias and exponentiate
				output += vector.weights[vector.weights.length-1][0];
				classProbs[classIndex]=Math.exp(output);
			}
			//make a prediction
			int prediction = 0;
			for(int index=1;index<classProbs.length;index++) {
				if(classProbs[index]>classProbs[prediction]) {
					prediction=index;
				}
			}
			predictions[sampleIndex]=weightVectors.get(prediction).classID;
		}
		return predictions;
	}
	
	/*
	 * A method to calculate the change to the weight update
	 */
	public static double calcWeightUpdate(String realClass, String predictClass, double classProb, double featureVal, boolean print) {
		double kroneckerDelta;
		if(realClass.equals(predictClass)) {
			kroneckerDelta = 1.0;
		}
		else {
			kroneckerDelta = 0.0;
		}
		if(print) {
			System.out.println("Gradient, class "+ realClass+": " + (kroneckerDelta-classProb));
		}
		return (kroneckerDelta-classProb)*featureVal;
	}
}

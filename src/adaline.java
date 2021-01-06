import java.util.*;

public class adaline {
	public ArrayList<weightVector> weightVectors;
	
	adaline(){
		this.weightVectors = new ArrayList<weightVector>();
	}
	
	/*
	 * Main training method via gradient descent
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
		
		int iterations = 0;
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
			if(prevMisclassifications != Integer.MAX_VALUE) {
				prevMisclassifications=misclassifications;
			}
			misclassifications=0;
			
			//main loop
			for(classificationSample sample: trainingData) {
				double[] neuronOutput = new double[weightVectors.size()];
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
					//add bias
					neuronOutput[classIndex]=output+vector.weights[vector.weights.length-1][0];
				}
				
				//update the update values
				for(int classIndex=0;classIndex<weightUpdates.size();classIndex++) {
					for(int feature=0;feature<sample.features.length;feature++) {
						boolean print = false;
						if(printOutput && feature==0) {
							print = true;
						}
						if(categoricalGuide[feature]==true) {
							weightUpdates.get(classIndex)[feature][(int)sample.features[feature]] += (double) calcWeightUpdate(sample.classification,weightVectors.get(classIndex).classID,neuronOutput[classIndex],1,print)/trainingData.size();
						}
						else {
							weightUpdates.get(classIndex)[feature][0] += (double) calcWeightUpdate(sample.classification,weightVectors.get(classIndex).classID,neuronOutput[classIndex],sample.features[feature],print)/trainingData.size();
						}
					}
					//update bias
					weightUpdates.get(classIndex)[weightUpdates.get(classIndex).length-1][0] += (double) calcWeightUpdate(sample.classification,weightVectors.get(classIndex).classID,neuronOutput[classIndex],1,false)/trainingData.size();
				}
				//make a prediction for the sake of tracking misclassifications
				int prediction = 0;
				for(int index=1;index<neuronOutput.length;index++) {
					if(neuronOutput[index]>neuronOutput[prediction]) {
						prediction=index;
					}
				}
				//check prediction
				if(!weightVectors.get(prediction).classID.equals(sample.classification)) {
					misclassifications++;
				}
			}
			//if we're moving uphill, end training
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
	 * Testing method, returns a class prediction based on the neuron with the highest positive output
	 */
	public String[] test(ArrayList<classificationSample> samples, boolean printOutput) {
		String[] predictions = new String[samples.size()];
		for(int sampleIndex=0;sampleIndex<samples.size();sampleIndex++) {
			classificationSample sample = samples.get(sampleIndex);
			double[] neuronOutput = new double[weightVectors.size()];
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
				//add bias
				neuronOutput[classIndex]=output + vector.weights[vector.weights.length-1][0];
			}
			//make a prediction
			int prediction = 0;
			for(int index=1;index<neuronOutput.length;index++) {
				if(neuronOutput[index]>neuronOutput[prediction]) {
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
	public static double calcWeightUpdate(String realClass, String predictClass, double neuronOutput, double featureVal, boolean print) {
		double expectedOutput;
		if(realClass.equals(predictClass)) {
			expectedOutput=1.0;
		}
		else {
			expectedOutput=-1.0;
		}
		if(print) {
			System.out.println("Gradient, class "+ realClass+": " + (expectedOutput-neuronOutput));
		}
		return (expectedOutput-neuronOutput)*featureVal;
	}
}

import java.util.*;
import java.io.*;

public class readData {
	
	public static ArrayList<classificationSample> readClassificationData(String file) throws Exception{
		ArrayList<String[]> stringArray = readToStringArray(file);
		switch(file) {
		case "breast-cancer-wisconsin.data":
			return parseBrca(stringArray);
		case "glass.data":
			return parseGlass(stringArray);
		case "house-votes-84.data":
			return parseVotes(stringArray);
		case "iris.data":
			return parseIris(stringArray);
		case "soybean-small.data":
			return parseSoybean(stringArray);
		default:
			throw new IllegalArgumentException("Unrecognized classification file: " + file);
		}
	}
	
	public static ArrayList<String[]> readToStringArray(String file) throws Exception{
		BufferedReader buffReader = null;
		ArrayList<String[]> stringArray;
		try {
			String currentLine;
			stringArray = new ArrayList<String[]>();
			buffReader = new BufferedReader(new FileReader(file));
			//Read file to string array
			while((currentLine=buffReader.readLine()) != null) {	//read until end of file
				if(!currentLine.trim().equals("")) {	//don't read empty lines
					stringArray.add(currentLine.split(","));
				}
			}
		}
		finally {
			if(buffReader != null) {buffReader.close();}
		}
		return stringArray;
	}
	
	
	/*
	 * The following are methods for taking input ArrayList<String[]> datasets and parsing them to create ArrayList<sampleNode> arrays
	 * This accomplishes separating classes/regression targets into a named variable and features into a named array, as well as parsing all numerical 
	 * features/regression targets into doubles. 
	 * 
	 * All categorical features have been transformed to an integer (which then must be transformed to a double)
	 */
	public static ArrayList<classificationSample> parseBrca(ArrayList<String[]> data){
		ArrayList<classificationSample> parsedData = new ArrayList<classificationSample>();
		for(String[] sample: data) {
			classificationSample newNode;
			
			if(sample[sample.length-1].equals("2")) {
				newNode = new classificationSample("NORMAL",new double[sample.length-2]);
			}
			else {
				newNode = new classificationSample("MALIGNANT",new double[sample.length-2]);
			}
			
			for(int feature=1;feature<sample.length-1;feature++) {
				if(sample[feature].equals("?")) {
					sample = helpers.randomImpute(data,sample,sample.length-1,feature);
				}
				newNode.features[feature-1] = Integer.parseInt(sample[feature]);
			}
			parsedData.add(newNode);
		}
		return parsedData;
	}
	
	public static ArrayList<classificationSample> parseIris(ArrayList<String[]> data){
		ArrayList<classificationSample> parsedData = new ArrayList<classificationSample>();
		for(String[] sample: data) {
			classificationSample newNode;
			
			newNode = new classificationSample(sample[sample.length-1],new double[sample.length-1]);
			
			for(int feature=0;feature<sample.length-1;feature++) {
				newNode.features[feature] = Double.parseDouble(sample[feature]);
			}
			parsedData.add(newNode);
		}
		return parsedData;
	}
	
	public static ArrayList<classificationSample> parseVotes(ArrayList<String[]> data){
		ArrayList<classificationSample> parsedData = new ArrayList<classificationSample>();
		for(String[] sample: data) {
			classificationSample newNode;
			
			newNode = new classificationSample(sample[0],new double[sample.length-1]);
			
			for(int feature=1;feature<sample.length;feature++) {
				switch(sample[feature]) {
				case "?":
					newNode.features[feature-1] = 0; break;
				case "n":
					newNode.features[feature-1] = 1; break;
				case "y":
					newNode.features[feature-1] = 2; break;
				default:
					newNode.features[feature-1] = 0; break;
				}
				
			}
			parsedData.add(newNode);
		}
		return parsedData;
	}
	
	public static ArrayList<classificationSample> parseGlass(ArrayList<String[]> data){
		ArrayList<classificationSample> parsedData = new ArrayList<classificationSample>();
		for(String[] sample: data) {
			classificationSample newNode;
			
			newNode = new classificationSample(sample[sample.length-1],new double[sample.length-2]);
			
			for(int feature=1;feature<sample.length-1;feature++) {
				newNode.features[feature-1] = Double.parseDouble(sample[feature]);
			}
			parsedData.add(newNode);
		}
		return parsedData;
	}
	
	public static ArrayList<classificationSample> parseSoybean(ArrayList<String[]> data){
		ArrayList<classificationSample> parsedData = new ArrayList<classificationSample>();
		for(String[] sample: data) {
			classificationSample newNode;
			
			newNode = new classificationSample(sample[sample.length-1],new double[sample.length-1]);
			
			for(int feature=0;feature<sample.length-1;feature++) {
				newNode.features[feature] = Double.parseDouble(sample[feature]);
			}
			parsedData.add(newNode);
		}
		return parsedData;
	}
}

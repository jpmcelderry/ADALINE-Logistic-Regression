import java.util.*;

public class partitionData {

	 public static ArrayList<ArrayList<classificationSample>> partitionClassificationData(ArrayList<classificationSample> data) {
		 
		 //Classification problems are shuffled, then sorted preserving the random order within each class
		 	Collections.shuffle(data);
		 	Collections.sort(data, new Comparator<classificationSample>() {
		 		public int compare(classificationSample row1, classificationSample row2) {
		 			return row1.classification.compareTo(row2.classification);
		 		}
		 	});
		 
		 	ArrayList<classificationSample> fold1 = new ArrayList<classificationSample>();
			 ArrayList<classificationSample> fold2 = new ArrayList<classificationSample>();
			 ArrayList<classificationSample> fold3 = new ArrayList<classificationSample>();
			 ArrayList<classificationSample> fold4 = new ArrayList<classificationSample>();
			 ArrayList<classificationSample> fold5 = new ArrayList<classificationSample>();
			 
			 //remove every 1 in 5 consecutive samples to each fold
			 for(int index=data.size()-1;index>=0;index--) {
				 switch(index%5) {
				 	case 0:
				 		fold1.add(data.remove(index));
				 		break;
				 	case 1:
				 		fold2.add(data.remove(index));
				 		break;
				 	case 2:
				 		fold3.add(data.remove(index));
				 		break;
				 	case 3:
				 		fold4.add(data.remove(index));
				 		break;
				 	case 4:
				 		fold5.add(data.remove(index));
				 		break;	
				 }
			 }
			 
			 ArrayList<ArrayList<classificationSample>> returnData = new ArrayList<ArrayList<classificationSample>>();
			 returnData.add(fold1);
			 returnData.add(fold2);
			 returnData.add(fold3);
			 returnData.add(fold4);
			 returnData.add(fold5);
			 
			 return returnData;
	 }
}

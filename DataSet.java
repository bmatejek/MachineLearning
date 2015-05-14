public class DataSet {
  //  list of DataPoints
  private DataPoint integer_data_points[];
  private DataPoint binary_data_points[];
  private String mapping[][];
  private String label_mapping[];
    
  // constructor for the data set
  public DataSet(String mapping[][], String label_mapping[], DataPoint data_points[]) {
    int n_binary_attributes = 0;                               // number of binary attributes    
    int n_classes_per_attribute[] = new int[mapping.length];   // number of classes per attribute
    for (int i = 0; i < n_classes_per_attribute.length; ++i) {
      n_classes_per_attribute[i] = 0;
    }
                                            
    // copy in the mapping array
    this.mapping = new String[mapping.length][];
    for (int i = 0; i < this.mapping.length; ++i) {
      this.mapping[i] = new String[mapping[i].length];
      for (int j = 0; j < this.mapping[i].length; ++j) {
        this.mapping[i][j] = mapping[i][j];
        // do not consider the null or the first (attr. name) elements
        if (mapping[i][j] != null && j != 0) {
          n_binary_attributes++;
          n_classes_per_attribute[i]++;
        }
      }
    }
    
    // copy in the label information
    this.label_mapping = new String[label_mapping.length];
    for (int i = 0; i < this.label_mapping.length; ++i) 
      this.label_mapping[i] = label_mapping[i];
    
    // create new vectors for integer and binary data points
    integer_data_points = new DataPoint[data_points.length];
    binary_data_points  = new DataPoint[data_points.length];
    
    // fill in the integer data points list
    for (int i = 0; i < integer_data_points.length; ++i) {
      integer_data_points[i] = new DataPoint(data_points[i]);
      
      // create an integer data point
      DataPoint data_point = integer_data_points[i];
      int label = data_point.Label();
      int attributes[] = new int[n_binary_attributes];
      
      // JAVA AUTOINITIALIZES TO ZERO
      // initialize all to 0
      // for (int j = 0; j < attributes.length; ++j) {
      //   attributes[j] = 0;
      // }
      
      // fill in the rest of the binary data set
      int seen_so_far = 0;
      for (int k = 0; k < this.mapping.length; ++k) {
        attributes[seen_so_far + data_point.KthAttribute(k)] = 1;
        seen_so_far += n_classes_per_attribute[k];
      }
      binary_data_points[i] = new DataPoint(attributes, label, true);
    }
  } 

  // functions to add in data points
  public DataPoint KthDataPoint(int k) {
    if (k >= integer_data_points.length || k < 0) {
      System.err.printf("Error: accessed illegal element %d.\n", k);
      return null;
    }
    return new DataPoint(integer_data_points[k]);
  }
  
  // return kth binary datapoint
  public DataPoint KthBinaryDataPoint(int k) {
    if (k >= binary_data_points.length || k < 0) {
      System.err.printf("Error: accessed illegal element %d.\n", k);
      return null;
    }
    return new DataPoint(binary_data_points[k]);
  }

  // return number of datapoints
  public int NDataPoints() {
    return integer_data_points.length;
  }
  
  // print out the data set
  public String toString() {
    String output = "";
    for (int i = 0; i < mapping.length; ++i) {
      output += mapping[i][0] + "\t";
    }
    output += "Label\n";
    for (int i = 0; i < integer_data_points.length; ++i) {
      for (int k = 0; k < integer_data_points[i].NAttributes(); ++k) {
        output += mapping[k][integer_data_points[i].KthAttribute(k) + 1] + "\t";
      }
      output += label_mapping[integer_data_points[i].Label()];
      output += "\n";
    }
    // return the output string
    return output;
  }
}
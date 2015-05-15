package MachineLearning.DecisionTree;

import java.util.Vector;
import java.util.HashMap;
import java.util.Arrays;
import java.util.List;
import MachineLearning.*;


public class DecisionTree implements Learner {
  private final boolean print_verbose;
  private final DataSet train;
  private Tree tree;
  
  // create the actual tree structure
  public class Tree { 
    int label;
    int splitting_attribute;
    HashMap<Integer, Tree> subtrees;
    
    // tree that is a leaf has no root test
    public Tree(int label) {
      this.label = label;
      this.splitting_attribute = -1;
      this.subtrees = null;
    }
    
    public Tree(int splitting_attribute, boolean throwaway) {
      this.splitting_attribute = splitting_attribute;
      this.label = -1;
      subtrees = new HashMap<Integer, Tree>();
    }
    
    // add the subtree to this tree
    public void AddSubTree(int key, Tree tree) {
      subtrees.put(key, tree);
    }
    
    // return the subtree 
    public Tree SubTree(int key) {
      return subtrees.get(key);
    }
    
    // print the tree
    public String toString() {
      return "";
    } 
  }
  
  // see if all of the examples have the same label
  private boolean SameClassification(Vector<DataPoint> examples) {
    int label = examples.get(0).Label();
    for (int i = 1; i < examples.size(); ++i)
      if (examples.get(i).Label() != label) return false;
    return true;
  }
  
  // return the largest example in the parent class
  private int PluralityValue(Vector<DataPoint> parent_examples) {
    int[] noccurences = new int[train.NLabels()];
    for (int i = 0; i < parent_examples.size(); ++i)
      noccurences[parent_examples.get(i).Label()]++;
    int maxoccurences = 0;
    int label = 0;
    for (int i = 0; i < noccurences.length; ++i) {
      if (maxoccurences < noccurences[i]) {
        maxoccurences = noccurences[i];
        label = i;
      }
    }
    return label;
  }
  
  private double log2(double q) {
    return Math.log(q) / Math.log(2);
  }
  
  private double B(double q) {
    if (q == 0.0 || q == 1.0) return 0.0;
    else return -1.0 * (q * log2(q) + (1 - q) * log2(1 - q));
  }
  
  private double Remainder(int A, Vector<DataPoint> examples) {
    // get the total number of positive and negative examples
    int p = 0; 
    int n = 0;
    for (int i = 0; i < examples.size(); ++i) {
      if (examples.get(i).Label() == 1) p++;
      else n++;
    }
    
    double remainder = 0.0;
    // go through every possible attribute value
    for (int k = 0; k < train.NClassesKthAttribute(A); ++k) {
      // keep track of the number of positive and negative examples for this attribute
      int pk = 0;
      int nk = 0;
      // go through each example
      for (int i = 0; i < examples.size(); ++i) {
        if (examples.get(i).KthAttribute(A) == k) {
          if (examples.get(i).Label() == 1) pk++;
          else nk++;
        }
      }
      if (pk != 0.0 || nk != 0.0) remainder += ((double) (pk + nk)) / ((double) (p + n)) * B(((double) pk) / ((double) (pk + nk)));
    }
    return remainder;
  }
  
  private int Importance(Vector<Integer> attributes, Vector<DataPoint> examples) {
    // return an array of size one for decision trees
    int A = -1;
    double least_remainder = Double.MAX_VALUE;
    for (int i = 0; i < attributes.size(); ++i) {
      int a = attributes.get(i);
      double remainder = Remainder(a, examples);
      if (remainder < least_remainder) {
        least_remainder = remainder;
        A = a;
      }
    }
    return A;
  }
  
  private Tree DecisionTreeLearning(Vector<DataPoint> examples,  Vector<Integer> attributes, Vector<DataPoint> parent_examples) {
    // if examples is empty then return Plurality-Value(parent_examples)
    if (examples.size() == 0) return new Tree(PluralityValue(parent_examples));
    // else if all examples have the same classification then return the classification
    else if (SameClassification(examples)) return new Tree(examples.get(0).Label());
    // else if attributes is empty then return Plurality-Value(examples)
    else if (attributes.size() == 0) return new Tree(PluralityValue(examples));
    // or else
    else {
      // A <- argmax a \in attributes Importance(a, examples)
      int A = Importance(attributes, examples);
      //System.out.println("Attribute: " + A);
      Tree tree = new Tree(A, true);
      
      // create a new set of attributes
      Vector<Integer> attributes_sans_A = new Vector<Integer>();
      for (int a = 0; a < attributes.size(); ++a)  {
        if (attributes.get(a) != A) attributes_sans_A.addElement(attributes.get(a));
      }
      //System.out.println("Size: " + attributes_sans_A.size());
      for (int vk = 0; vk < train.NClassesKthAttribute(A); ++vk) {
        // remove all of the examples that are not of this class
        Vector<DataPoint> exs = new Vector<DataPoint>();
        for (int e = 0; e < examples.size(); ++e) {
          DataPoint data_point = examples.get(e);
          if (data_point.KthAttribute(A) == vk)
            exs.addElement(new DataPoint(data_point));
        }
        //System.out.println("AttributeLabel: " + vk);
        //System.out.println("Examples size: " + exs.size());
        tree.AddSubTree(vk, DecisionTreeLearning(exs, attributes_sans_A, examples));
      }
      
      return tree;
    }
  }
  
  
  // constructor for DecisionTree
  public DecisionTree(DataSet train, boolean print_verbose) { 
    this.print_verbose = print_verbose;
    this.train = train;
    
    Vector<DataPoint> parent_examples = new Vector<DataPoint>();
    Vector<DataPoint> examples = new Vector<DataPoint>();
    for (int i = 0; i < train.NDataPoints(); ++i) {
      examples.addElement(train.KthDataPoint(i));
    }
    Vector<Integer> attributes = new Vector<Integer>();
    for (int i = 0; i < train.NAttributes(); ++i) {
      attributes.addElement(i);
    }
    tree = DecisionTreeLearning(examples, attributes, parent_examples);
  }

  // classify example
  private int ClassifyExample(DataPoint data_point, Tree tree) {
    if (tree.label != -1) return tree.label;
    else {
      int splitting_value = data_point.KthAttribute(tree.splitting_attribute);
      return ClassifyExample(data_point, tree.SubTree(splitting_value));
    }
  }
  
  // classify a particular data
  public int[] Classify(DataSet test) {
    int[] labels = new int[test.NDataPoints()];
    
    for (int i = 0; i < test.NDataPoints(); ++i) {
      labels[i] = ClassifyExample(test.KthDataPoint(i), tree);
    }
    
    return labels;
  }
}
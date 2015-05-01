public class Label {
  // name of the class
  private String name;
  // id of this class
  private int id;
  
  // constructor for the label
  public Label(String name, int id) {
    this.name = name;
    this.id = id;
  }
  
  // return the name of the label
  public String Name() {
    return name;
  }
  
  // return the id of the label
  public int ID() {
    return id;
  }
}
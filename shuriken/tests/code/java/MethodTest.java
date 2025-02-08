// MethodTest.java
public class MethodTest {
    private int value;

    public MethodTest() {
        value = 42;
    }

    private void privateMethod() {
        value++;
    }

    public void publicMethod() {
        privateMethod();
        System.out.println(value);
    }
}
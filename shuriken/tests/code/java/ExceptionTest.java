public class ExceptionTest {
    private int value;

    public void methodWithTryCatch() {
        try {
            riskyOperation();
            value = 42;
        } catch (IllegalArgumentException e) {
            System.out.println("Caught IllegalArgumentException");
            value = -1;
        } catch (Exception e) {
            System.out.println("Caught general exception");
            value = -2;
        }
    }

    private void riskyOperation() throws IllegalArgumentException {
        if (value == 0) {
            throw new IllegalArgumentException("Value cannot be zero");
        }
        throw new RuntimeException("Other error");
    }

    // Method with nested try-catch for more complex basic block analysis
    public void methodWithNestedTryCatch() {
        try {
            try {
                value = Integer.parseInt("not a number");
            } catch (NumberFormatException e) {
                System.out.println("Inner catch");
                throw new IllegalStateException("Propagating error");
            }
        } catch (IllegalStateException e) {
            System.out.println("Outer catch");
        }
    }
}
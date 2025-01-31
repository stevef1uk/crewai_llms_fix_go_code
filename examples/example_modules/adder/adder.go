package adder

// Add adds two integers together.
func Add(a, b int) int {
	return a + b
}

func main() {
	// Add two integers.
	sum := Add(1, 2)

	// Print the sum.
	println(sum)
}
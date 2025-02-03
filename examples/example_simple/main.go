package main

import (
    "fmt"
    "strings"
)

// Add returns the sum of two integers
func Add(a, b int) int {
    return a + b
}

// Divide returns the division of two numbers and an error if divisor is zero
func Divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero is not allowed")
    }
    return a / b, nil
}

// ReverseString returns the reversed version of the input string
func ReverseString(s string) string {
    runes := []rune(s)
    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    return string(runes)
}

// FilterEvenNumbers returns a slice containing only even numbers from the input slice
func FilterEvenNumbers(numbers []int) []int {
    if numbers == nil {
        return []int{}
    }
    var result []int
    for _, n := range numbers {
        if n%2 == 0 {
            result = append(result, n)
        }
    }
    return result
}

// PersonInfo represents a person's basic information
type PersonInfo struct {
    FirstName string
    LastName  string
    Age       int
}

// GetFullName returns the full name of a person
func (p PersonInfo) GetFullName() string {
    return strings.TrimSpace(p.FirstName + " " + p.LastName)
}

// IsAdult returns true if the person is 18 or older
func (p PersonInfo) IsAdult() bool {
    return p.Age >= 18
}

// PrintGreeting prints a greeting message for the person
func (p PersonInfo) PrintGreeting() {
    fmt.Printf("Hello, %s! You are %d years old.\n", p.GetFullName(), p.Age)
}

func main() {
    // Using the Add function
    result := Add(5, 3)
    fmt.Printf("5 + 3 = %d\n", result)

    // Using the Divide function
    if quotient, err := Divide(10, 2); err == nil {
        fmt.Printf("10 / 2 = %.2f\n", quotient)
    }

    // Using the ReverseString function
    reversed := ReverseString("Hello, World!")
    fmt.Printf("Reversed string: %s\n", reversed)

    // Using FilterEvenNumbers
    numbers := []int{1, 2, 3, 4, 5, 6}
    evenNums := FilterEvenNumbers(numbers)
    fmt.Printf("Even numbers: %v\n", evenNums)

    // Using PersonInfo
    person := PersonInfo{
        FirstName: "John",
        LastName:  "Doe",
        Age:       25,
    }
    person.PrintGreeting()
}

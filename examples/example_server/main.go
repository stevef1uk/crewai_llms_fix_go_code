package main

import (
    "fmt"
    "net/http"
)

func getGreeting() string {
   return "Hello, World!"
}

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, getGreeting())
}

func main() {
    http.HandleFunc("/", handler)
    fmt.Println("Server starting on port 8080...")
    http.ListenAndServe(":8080", nil)
}

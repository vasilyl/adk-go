// Package main provides a simple agent that can interact with BigQuery.
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	texporter "github.com/GoogleCloudPlatform/opentelemetry-operations-go/exporter/trace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/plugin/agentanalytics"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

func initTracer(projectID string) (func(), error) {
	exporter, err := texporter.New(texporter.WithProjectID(projectID))
	if err != nil {
		return nil, err
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
	)
	otel.SetTracerProvider(tp)

	return func() {
		tp.Shutdown(context.Background())
	}, nil
}

const (
	// ColorReset resets the color.
	ColorReset = "\033[0m"
	// ColorRed sets the color to red.
	ColorRed = "\033[31m"
	// ColorYellow sets the color to yellow.
	ColorYellow = "\033[33m"
)

func main() {
	ctx := context.TODO()
	projectID := os.Getenv("PROJECT_ID")
	log.Printf("Project ID: %s", projectID)
	datasetID := os.Getenv("DATASET_ID")
	log.Printf("Dataset ID: %s", datasetID)
	tableID := os.Getenv("TABLE_ID")
	log.Printf("Table ID: %s", tableID)

	shutdownTracer, err := initTracer(projectID)
	if err != nil {
		log.Fatalf("Failed to setup tracing: %v", err)
	}
	defer shutdownTracer()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	temp := float32(0.5)
	topP := float32(0.9)

	a, err := llmagent.New(llmagent.Config{
		Name:        "my_bq_agent",
		Model:       model,
		Instruction: "You are a helpful assistant with access to BigQuery tools. You can also set session state using the `set_state` tool.",
		Tools:       []tool.Tool{},
		GenerateContentConfig: &genai.GenerateContentConfig{
			Temperature: &temp,
			TopP:        &topP,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	sessionService := session.InMemoryService()
	sessionResp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: "my_bq_agent",
		UserID:  "user1234",
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	bqLoggingPlugin, err := agentanalytics.NewBigQueryAgentAnalyticsPlugin(ctx, projectID, datasetID, tableID)
	if err != nil {
		log.Fatalf("Failed to create bq plugin: %v", err)
	}
	defer bqLoggingPlugin.Close()

	r, err := runner.New(runner.Config{
		AppName:        "my_bq_agent",
		Agent:          a,
		SessionService: sessionService,
		PluginConfig: runner.PluginConfig{
			Plugins: []*plugin.Plugin{bqLoggingPlugin},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print(ColorYellow + "User: " + ColorReset)
		userInput, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		userInput = strings.TrimSpace(userInput)
		if strings.ToLower(userInput) == "quit" {
			break
		}

		userMsg := genai.NewContentFromText(userInput, genai.RoleUser)

		fmt.Print(ColorRed + "Agent: " + ColorReset)
		var output strings.Builder
		for event, err := range r.Run(ctx, "user1234", sessionResp.Session.ID(), userMsg, agent.RunConfig{}) {
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				break
			}
			if event.IsFinalResponse() && event.LLMResponse.Content != nil {
				for _, p := range event.LLMResponse.Content.Parts {
					if p.Text != "" {
						output.WriteString(p.Text)
					}
				}
			}
		}
		if output.Len() > 0 {
			fmt.Print(ColorRed + output.String() + ColorReset)
		}
		fmt.Println()
	}

	fmt.Println("Closing connections...")
}

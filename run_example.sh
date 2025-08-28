#!/bin/bash

# Complete RAG Platform Example Runner
# ====================================

set -e  # Exit on any error

echo "🚀 RAG Platform Complete Example Runner"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check if required files exist
check_files() {
    print_status $BLUE "📁 Checking required files..."
    
    required_files=(
        "start_mcp_server.py"
        "complete_rag_example.py"
        "manage.py"
        ".env"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_status $RED "❌ Missing required file: $file"
            exit 1
        fi
    done
    
    print_status $GREEN "✅ All required files found"
}

# Check if .env has necessary configuration
check_env() {
    print_status $BLUE "🔧 Checking environment configuration..."
    
    if [ ! -f ".env" ]; then
        print_status $YELLOW "⚠️  No .env file found. Creating from template..."
        cp .env.example .env
        print_status $YELLOW "📝 Please edit .env with your API keys and run again."
        exit 1
    fi
    
    # Check for at least one API key
    if ! grep -q "OPENAI_API_KEY=sk-" .env && ! grep -q "ANTHROPIC_API_KEY=sk-ant-" .env; then
        print_status $YELLOW "⚠️  No API keys found in .env file."
        print_status $YELLOW "   Add at least one of: OPENAI_API_KEY or ANTHROPIC_API_KEY"
        print_status $YELLOW "   Example: OPENAI_API_KEY=sk-your-key-here"
        exit 1
    fi
    
    print_status $GREEN "✅ Environment configuration looks good"
}

# Start MCP server in background
start_mcp_server() {
    print_status $BLUE "🔌 Starting MCP server..."
    
    # Check if MCP server is already running
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        print_status $GREEN "✅ MCP server is already running"
        return 0
    fi
    
    # Start MCP server in background
    python start_mcp_server.py > mcp_server.log 2>&1 &
    MCP_PID=$!
    echo $MCP_PID > mcp_server.pid
    
    print_status $YELLOW "⏳ Waiting for MCP server to start..."
    
    # Wait for MCP server to be ready (max 30 seconds)
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            print_status $GREEN "✅ MCP server is running (PID: $MCP_PID)"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    
    print_status $RED "❌ MCP server failed to start within 30 seconds"
    print_status $YELLOW "   Check mcp_server.log for details"
    return 1
}

# Start Django server in background
start_django_server() {
    print_status $BLUE "🌐 Starting Django server..."
    
    # Check if Django server is already running
    if curl -s http://localhost:8000/api/rag/ping/ > /dev/null 2>&1; then
        print_status $GREEN "✅ Django server is already running"
        return 0
    fi
    
    # Run migrations first
    print_status $BLUE "📊 Running database migrations..."
    python manage.py migrate > django_migrate.log 2>&1
    
    # Start Django server in background
    python manage.py runserver > django_server.log 2>&1 &
    DJANGO_PID=$!
    echo $DJANGO_PID > django_server.pid
    
    print_status $YELLOW "⏳ Waiting for Django server to start..."
    
    # Wait for Django server to be ready (max 30 seconds)
    for i in {1..30}; do
        if curl -s http://localhost:8000/api/rag/ping/ > /dev/null 2>&1; then
            print_status $GREEN "✅ Django server is running (PID: $DJANGO_PID)"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    
    print_status $RED "❌ Django server failed to start within 30 seconds"
    print_status $YELLOW "   Check django_server.log for details"
    return 1
}

# Run the complete example
run_example() {
    print_status $BLUE "🎯 Running complete RAG example..."
    
    # Give services a moment to fully initialize
    sleep 3
    
    if python complete_rag_example.py --skip-check; then
        print_status $GREEN "🎉 Example completed successfully!"
        return 0
    else
        print_status $RED "❌ Example failed"
        return 1
    fi
}

# Cleanup function
cleanup() {
    print_status $BLUE "🧹 Cleaning up background processes..."
    
    # Stop Django server
    if [ -f django_server.pid ]; then
        DJANGO_PID=$(cat django_server.pid)
        if ps -p $DJANGO_PID > /dev/null 2>&1; then
            kill $DJANGO_PID
            print_status $YELLOW "🛑 Stopped Django server (PID: $DJANGO_PID)"
        fi
        rm -f django_server.pid
    fi
    
    # Stop MCP server
    if [ -f mcp_server.pid ]; then
        MCP_PID=$(cat mcp_server.pid)
        if ps -p $MCP_PID > /dev/null 2>&1; then
            kill $MCP_PID
            print_status $YELLOW "🛑 Stopped MCP server (PID: $MCP_PID)"
        fi
        rm -f mcp_server.pid
    fi
    
    print_status $GREEN "✅ Cleanup completed"
}

# Main execution
main() {
    print_status $GREEN "Starting RAG platform example..."
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run checks
    check_files
    check_env
    
    # Start services
    if ! start_mcp_server; then
        print_status $RED "❌ Failed to start MCP server"
        exit 1
    fi
    
    if ! start_django_server; then
        print_status $RED "❌ Failed to start Django server"
        exit 1
    fi
    
    # Run the example
    if run_example; then
        print_status $GREEN "🏁 All done! Example completed successfully."
        print_status $BLUE "📋 Check the logs for detailed output:"
        print_status $BLUE "   - MCP server: mcp_server.log"
        print_status $BLUE "   - Django server: django_server.log"
        print_status $BLUE "   - Migrations: django_migrate.log"
    else
        print_status $RED "❌ Example failed. Check the logs for details."
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --quick-test   Run quick test only"
        echo "  --cleanup      Just cleanup running processes"
        echo ""
        echo "This script will:"
        echo "1. Check environment setup"
        echo "2. Start MCP server and Django server"
        echo "3. Run the complete RAG example"
        echo "4. Cleanup processes when done"
        exit 0
        ;;
    --quick-test)
        print_status $BLUE "🧪 Running quick test only..."
        python quick_test.py
        exit $?
        ;;
    --cleanup)
        cleanup
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_status $RED "❌ Unknown option: $1"
        print_status $BLUE "Use --help for usage information"
        exit 1
        ;;
esac
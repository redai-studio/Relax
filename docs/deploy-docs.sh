#!/bin/bash
# Deploy documentation to GitHub Pages or other hosting services

set -e

echo "🚀 Deploying Relax Documentation..."

# Navigate to docs directory
cd "$(dirname "$0")/../docs"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Build documentation
echo "🔨 Building documentation..."
npm run docs:build

# Check if build was successful
if [ ! -d ".vitepress/dist" ]; then
    echo "❌ Build failed! .vitepress/dist directory not found."
    exit 1
fi

echo "✅ Documentation built successfully!"

# Deployment options
if [ "$1" == "github" ]; then
    echo "📤 Deploying to GitHub Pages..."
    
    cd .vitepress/dist
    
    # Initialize git if needed
    if [ ! -d ".git" ]; then
        git init
        git add -A
        git commit -m "Deploy documentation"
        git branch -M gh-pages
    fi
    
    # Push to GitHub Pages
    git push -f git@github.com:redai-studio/Relax.git gh-pages
    
    echo "✅ Deployed to GitHub Pages!"
    
elif [ "$1" == "vercel" ]; then
    echo "📤 Deploying to Vercel..."
    npx vercel --prod
    echo "✅ Deployed to Vercel!"
    
elif [ "$1" == "netlify" ]; then
    echo "📤 Deploying to Netlify..."
    npx netlify deploy --prod --dir=.vitepress/dist
    echo "✅ Deployed to Netlify!"
    
else
    echo "📋 Build complete! Distribution files are in .vitepress/dist"
    echo ""
    echo "To deploy, run:"
    echo "  ./scripts/deploy-docs.sh github   # Deploy to GitHub Pages"
    echo "  ./scripts/deploy-docs.sh vercel   # Deploy to Vercel"
    echo "  ./scripts/deploy-docs.sh netlify  # Deploy to Netlify"
    echo ""
    echo "Or manually upload the .vitepress/dist directory to your hosting service."
fi

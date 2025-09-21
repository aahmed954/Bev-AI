FROM node:18-alpine

WORKDIR /app

# Install basic dependencies
RUN npm install express cytoscape cytoscape-dagre cytoscape-cose-bilkent

# Create a simple cytoscape server
COPY package.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["node", "server.js"]
# proj-ui/Dockerfile

# --- Build Stage ---
FROM node:20-alpine AS build
WORKDIR /app
COPY server/package*.json ./
COPY server/tsconfig*.json ./
COPY server/src ./src
COPY server/vite.config.ts ./
COPY server/index.html ./
COPY server/public ./public
RUN npm install ci

# Allow overriding API URL at build time:
# docker build --build-arg VITE_API_URL=http://localhost:8000
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# --- Run Stage ---
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
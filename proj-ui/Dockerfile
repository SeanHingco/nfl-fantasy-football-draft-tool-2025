# proj-ui/Dockerfile

# --- Build Stage ---
FROM node:20-alpine AS build
WORKDIR /app
COPY proj-ui/package*.json ./
COPY proj-ui/tsconfig*.json ./
COPY proj-ui/src ./src
COPY proj-ui/vite.config.ts ./
COPY proj-ui/index.html ./
COPY proj-ui/public ./public
RUN npm install ci

# Allow overriding API URL at build time:
# docker build --build-arg VITE_API_URL=http://localhost:8000
ARG VITE_API_BASE
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build

# --- Run Stage ---
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
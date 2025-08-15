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
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# --- Run Stage ---
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY proj-ui/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
RUN nginx -t
CMD ["nginx", "-g", "daemon off;"]
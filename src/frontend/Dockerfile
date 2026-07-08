# pick Node.js runtime as the base image
FROM node:22-alpine AS build

# set the working directory in the container
WORKDIR /app

# copy package.json to working directory
COPY package.json package-lock.json ./

# install project dependencies with version in package lock
RUN npm ci

# copy the entire React app code to the container
COPY . .

# compile app
RUN npm run build

# Stage 1, based on Nginx, to have only the compiled app, ready for production with Nginx
FROM nginx:alpine

COPY --from=build /app/dist/ /usr/share/nginx/html

# nginx image's startup script looks here for templates
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
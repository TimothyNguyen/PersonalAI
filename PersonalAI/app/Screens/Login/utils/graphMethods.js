// me?fields=id,name,birthday,movies,music,feed,friends,television,books,favorite_athletes,email,languages,likes,family,games

import {
  GraphRequest,
  GraphRequestManager
} from 'react-native-fbsdk';

const makeSingleGraphRequest = (request) => {
  return new GraphRequestManager().addRequest(request).start();
}

export const getUserId = (callback) => {
  const request = new GraphRequest('/me/id', null, callback);
  makeSingleGraphRequest(request);
}

export const getName = (callback) => {
  const request = new GraphRequest('/me/name', null, callback);
  makeSingleGraphRequest(request);
}

export const getBirthday = (callback) => {
  const request = new GraphRequest('/me/birthday', null, callback);
  makeSingleGraphRequest(request);
}

// Get Other User data more important

export const getFeed = (callback) => {
  const request = new GraphRequest('/me/feed', null, callback);
  makeSingleGraphRequest(request);
}

export const getFriends = (callback) => {
  const request = new GraphRequest('/me/friends', null, callback);
  makeSingleGraphRequest(request);
}

export const getMovies = (callback) => {
  const request = new GraphRequest('/me/movies', null, callback);
  makeSingleGraphRequest(request);
}

import { View, Text } from 'react-native';
import LottieView from 'lottie-react-native';

const Loader = () => {
	return (
		<LottieView
			source={require('../assets/animations/loading.json')}
			autoPlay
			loop
		/>
	);
};

export default Loader;

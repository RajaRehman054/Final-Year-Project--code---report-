import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import Login from './screens/Login';
import DashboardLayout from './screens/DashboardLayout';
import JobPost from './screens/MainScreens/JobPost';
import Bids from './screens/MainScreens/Bids';
import Chat from './screens/MainScreens/Chat';
import NewsFeed from './screens/MainScreens/NewsFeed';
import TrendingBids from './screens/SecondaryScreens/TrendingBids';
import ActiveBids from './screens/SecondaryScreens/ActiveBids';
import ContentDetails from './screens/SecondaryScreens/ContentDetails';
import Search from './screens/MainScreens/Search';
import JobDetails from './screens/SecondaryScreens/JobDetails';
import MyProfile from './screens/SecondaryScreens/MyProfile';
import OthersProfile from './screens/SecondaryScreens/OthersProfile';
import Settings from './screens/MainScreens/Settings';
import Payment from './screens/MainScreens/Payment';
import CheckoutSuccess from './screens/SecondaryScreens/CheckoutSuccess';
import Auth from './Components/Auth';
import { store } from './redux/store';
import { Provider } from 'react-redux';

const router = createBrowserRouter([
	{
		path: '/',
		element: <Auth />,
		children: [
			{
				path: '/login',
				element: <Login />,
			},
			{
				path: '/',
				element: <DashboardLayout />,
				children: [
					{
						path: '/',
						element: <NewsFeed />,
					},
					{
						path: '/myProfile',
						element: <MyProfile />,
					},
					{
						path: '/otherProfile',
						element: <OthersProfile />,
					},
					{
						path: '/search',
						element: <Search />,
					},
					{
						path: '/JobPost',
						element: <JobPost />,
					},
					{
						path: '/JobDetails',
						element: <JobDetails />,
					},
					{
						path: '/bids',
						element: <Bids />,
					},
					{
						path: '/trendingsBids',
						element: <TrendingBids />,
					},
					{
						path: '/activeBids',
						element: <ActiveBids />,
					},
					{
						path: '/chat',
						element: <Chat />,
					},
					{
						path: '/contentDetails',
						element: <ContentDetails />,
					},
					{
						path: '/payment',
						element: <Payment />,
					},
					{
						path: '/settings',
						element: <Settings />,
					},
					{
						path: '/checkoutsuccess',
						element: <CheckoutSuccess />,
					},
				],
			},
		],
	},
]);

export default function App() {
	return (
		<Provider store={store}>
			<RouterProvider router={router} />
		</Provider>
	);
}
